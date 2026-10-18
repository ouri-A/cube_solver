"""
3D view of the scanned cube that starts the solve and follows it move by move.

It draws the latest ScanResult from scan_result in the colours the camera read. SPACE
sends the solution to the executor as an Execute goal. From then on the drawn cube only
moves when executed_move reports a move done, so it shows what the cube really did:
the stand-in executor now, the robot later.

Two threads, kept apart on purpose. ROS callbacks run on a background thread and only
put events on a queue. Ursina runs on the main thread and takes them off the queue in
its per-frame update(), in arrival order. A SPACE press goes the other way through a
second queue that a ROS timer empties, so every rclpy call stays on the ROS thread and
every Ursina call on the main one.
"""

import queue
import threading

from action_msgs.msg import GoalStatus
from cube_interfaces.action import Execute
from cube_interfaces.msg import ExecutedMove, ScanResult
from cube_solver import sim_view
from cube_solver.cube_model import FACES
import rclpy
from rclpy.action import ActionClient
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)


class ViewerNode(Node):
    """The ROS side: subscriptions and the action client. Never touches Ursina."""

    def __init__(self, events):
        super().__init__('viewer')
        self._events = events            # to the Ursina thread
        self._requests = queue.Queue()   # from the Ursina thread: move lists to execute
        self.create_subscription(ScanResult, 'scan_result', self.on_scan, LATCHED)
        self.create_subscription(ExecutedMove, 'executed_move', self.on_move, 10)
        self._execute = ActionClient(self, Execute, 'execute')
        self.create_timer(0.05, self.send_requested)

    def on_scan(self, msg):
        self._events.put(('scan', msg))

    def on_move(self, msg):
        self._events.put(('move', msg))

    def request_solve(self, moves):
        """Ask for a goal from any thread; the ROS timer below sends it."""
        self._requests.put(list(moves))

    def send_requested(self):
        try:
            moves = self._requests.get_nowait()
        except queue.Empty:
            return
        if not self._execute.server_is_ready():
            self._events.put(('status', 'no executor running'))
            return
        future = self._execute.send_goal_async(Execute.Goal(moves=moves))
        future.add_done_callback(self.on_goal_response)

    def on_goal_response(self, future):
        handle = future.result()
        if not handle.accepted:
            self._events.put(('status', 'executor refused the goal'))
            return
        self._events.put(('status', 'executing'))
        handle.get_result_async().add_done_callback(self.on_result)

    def on_result(self, future):
        response = future.result()
        if response.status == GoalStatus.STATUS_SUCCEEDED:
            self._events.put(('status', 'done'))
        else:
            self._events.put(('status', f'stopped after {response.result.done} moves: '
                                        f'{response.result.message or response.status}'))


class MovesLine:
    """Player's text line without its key hints: here SPACE sends the solve instead."""

    def __init__(self, text):
        self._text = text

    @property
    def text(self):
        return self._text.text

    @text.setter
    def text(self, value):
        self._text.text = value.split('\n')[0]


class Scene:
    """The Ursina side: one drawn cube, rebuilt for each new scan. Main thread only."""

    def __init__(self, node, events):
        from ursina import Entity, Text, window
        self._node, self._events = node, events
        self.app = sim_view.make_app('cube_solver viewer')
        self._hud = MovesLine(sim_view.add_camera_and_title('waiting for a scan'))
        self._status = Text('', position=window.bottom_left + (0.02, 0.15), scale=0.9)
        self._cubies, self._player, self._executing = {}, None, False
        Entity(update=self.update, input=self.on_key)

    def update(self):
        """Every frame: apply what ROS sent since the last one, oldest first."""
        while True:
            try:
                kind, payload = self._events.get_nowait()
            except queue.Empty:
                return
            if kind == 'scan':
                self.show_scan(payload)
            elif kind == 'move':
                self.follow(payload)
            else:
                self._executing = payload == 'executing'
                self._status.text = payload

    def show_scan(self, msg):
        if self._executing:
            self._status.text = 'new scan ignored: a solve is running'
            return
        from ursina import destroy
        for cubie in self._cubies.values():
            destroy(cubie)
        colours = {f: (c.r, c.g, c.b) for f, c in zip(FACES, msg.colours)}
        self._cubies = sim_view.build(msg.state, colours)
        self._player = sim_view.Player(self._cubies, msg.state, ' '.join(msg.solution), self._hud)
        self._status.text = 'space: solve' if msg.solution else 'already solved'

    def follow(self, msg):
        if self._player is None:
            return
        try:
            self._player.follow(msg.move)
        except ValueError as e:
            self._status.text = f'out of step with the executor: {e}'

    def on_key(self, key):
        if key == 'space' and self._player is not None and not self._executing:
            if self._player.done == 0 and self._player.moves:
                self._status.text = 'sending...'
                self._node.request_solve(self._player.moves)


def spin(executor):
    """Spin on a background thread until the context shuts down."""
    try:
        executor.spin()
    except ExternalShutdownException:
        pass


def main(args=None):
    rclpy.init(args=args)
    events = queue.Queue()
    node = ViewerNode(events)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=spin, args=(executor,), daemon=True)
    spinner.start()
    try:
        Scene(node, events).app.run()
    except (KeyboardInterrupt, ExternalShutdownException, SystemExit):
        pass
    finally:
        # Shut the context down first: that wakes the spin thread and makes it leave, so
        # nothing below tears down what it is still waiting on.
        rclpy.try_shutdown()
        spinner.join()
        executor.shutdown()
        node.destroy_node()
