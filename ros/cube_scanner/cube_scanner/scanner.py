"""
Guided webcam scan that publishes each solvable cube it reads.

The scan window is the one from cube_solver.scan, unchanged: same guide, same keys, same
scans/ folder. The only difference is what happens when the sixth face is read. Instead
of printing the result and opening the viewer itself, this node publishes it as a
ScanResult on scan_result. The topic is transient local, so a viewer that starts after
the scan still gets the latest one.

Parameters
----------
camera : int, default 0
    /dev/videoN to scan with.
mirror : bool, default False
    The camera delivers a mirrored image.

"""

import threading

from cube_interfaces.msg import ScanResult
from cube_solver import scan
from cube_solver.cube_model import FACES
import cv2
import rclpy
from rclpy.executors import ExternalShutdownException, SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA

LATCHED = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL)


def to_message(result, stamp):
    """Turn a finished vision scan into a ScanResult; None if the read did not solve."""
    if result.solution is None:
        return None
    msg = ScanResult()
    msg.header.stamp = stamp
    msg.state = result.resolution.state
    msg.colours = [ColorRGBA(r=float(r), g=float(g), b=float(b), a=1.0)
                   for r, g, b in (result.colours[f] for f in FACES)]
    msg.solution = [] if result.solution == '(already solved)' else result.solution.split()
    return msg


class Scanner(Node):
    """Owns the scan_result publisher; the scan window itself runs on the main thread."""

    def __init__(self):
        super().__init__('scanner')
        self.declare_parameter('camera', 0)
        self.declare_parameter('mirror', False)
        self._results = self.create_publisher(ScanResult, 'scan_result', LATCHED)

    def publish_result(self, result):
        """Publish a scan that solved; log why one that did not was dropped."""
        msg = to_message(result, self.get_clock().now().to_msg())
        if msg is None:
            self.get_logger().warning(f'scan not published: {result.error}  ({result.folder})')
            return
        self._results.publish(msg)
        self.get_logger().info(
            f'published scan {result.folder.name}: {len(msg.solution)} moves to solve')

    def scan(self):
        """Run the scan window until ESC; returns its exit code."""
        index = self.get_parameter('camera').value
        cam = cv2.VideoCapture(index)
        if not cam.isOpened():
            self.get_logger().error(f'could not open camera {index}')
            return 1
        try:
            return scan.run(cam, self.get_parameter('mirror').value, None, self.publish_result)
        finally:
            cam.release()
            cv2.destroyAllWindows()


def spin(executor):
    """Spin on a background thread until the context shuts down."""
    try:
        executor.spin()
    except ExternalShutdownException:
        pass


def main(args=None):
    rclpy.init(args=args)
    node = Scanner()
    # ROS work (parameter services, publishing) is served from a background thread, so the
    # OpenCV window keeps the main thread, which is where its GUI expects to run.
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spinner = threading.Thread(target=spin, args=(executor,), daemon=True)
    spinner.start()
    code = 0
    try:
        code = node.scan()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        # Shut the context down first: that wakes the spin thread and makes it leave, so
        # nothing below tears down what it is still waiting on.
        rclpy.try_shutdown()
        spinner.join()
        executor.shutdown()
        node.destroy_node()
    return code
