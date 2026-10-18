"""
Stand-in for the robot: carries out an Execute goal one move at a time.

It does what the robot node will do, minus the motors: accept a list of moves, wait
the time a turn would take, and report each move once it is done, on the action's
feedback for whoever sent the goal and on the executed_move topic for anyone else
(the viewer). The real robot node later takes the same goal and publishes the same
topic, so nothing that listens to this one has to change.

Parameters
----------
seconds_per_move : float, default 1.0
    How long each move takes.

"""

import threading
import time

from cube_interfaces.action import Execute
from cube_interfaces.msg import ExecutedMove
import rclpy
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.node import Node

from .moves import bad_moves


class MoveExecutor(Node):
    """Action server for Execute, one goal at a time."""

    def __init__(self):
        super().__init__('executor')
        self.declare_parameter('seconds_per_move', 1.0)
        self._busy = False
        self._lock = threading.Lock()
        self._moves = self.create_publisher(ExecutedMove, 'executed_move', 10)
        # Reentrant group + MultiThreadedExecutor, as in ros2/examples' minimal_action_server:
        # goal and cancel requests are still answered while a goal is executing.
        self._server = ActionServer(
            self, Execute, 'execute',
            execute_callback=self.execute,
            goal_callback=self.on_goal,
            cancel_callback=self.on_cancel,
            callback_group=ReentrantCallbackGroup())

    def on_goal(self, goal):
        """Accept a goal only if every move is a face turn and no other goal is running."""
        bad = bad_moves(goal.moves)
        if bad:
            self.get_logger().warning(f'rejected: not face turns: {bad}')
            return GoalResponse.REJECT
        with self._lock:
            if self._busy:
                self.get_logger().warning('rejected: already executing a goal')
                return GoalResponse.REJECT
            self._busy = True
        return GoalResponse.ACCEPT

    def on_cancel(self, goal_handle):
        """Allow cancelling; the running goal stops before its next move."""
        return CancelResponse.ACCEPT

    def execute(self, goal_handle):
        """Carry out the moves in order, reporting each one as it finishes."""
        moves = list(goal_handle.request.moves)
        pace = self.get_parameter('seconds_per_move').value
        result = Execute.Result()
        try:
            self.get_logger().info(f'executing {len(moves)} moves: {" ".join(moves)}')
            for index, move in enumerate(moves, start=1):
                if goal_handle.is_cancel_requested:
                    goal_handle.canceled()
                    result.message = f'cancelled after {result.done} moves'
                    return result
                time.sleep(pace)                 # the robot turns the face here
                self._moves.publish(ExecutedMove(index=index, move=move))
                goal_handle.publish_feedback(Execute.Feedback(done=index, last_move=move))
                result.done = index
            goal_handle.succeed()
            result.success = True
            return result
        finally:
            with self._lock:
                self._busy = False


def main(args=None):
    rclpy.init(args=args)
    node = MoveExecutor()
    try:
        rclpy.spin(node, executor=MultiThreadedExecutor())
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
