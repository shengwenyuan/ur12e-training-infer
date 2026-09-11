"""ROS 2 transport kept outside pure lifecycle, model and scheduler code."""

# Optional backend imports keep the robot/CLI dependency boundary lightweight.
# ROS packages are validated inside the Jazzy image.
# pylint: disable=import-outside-toplevel,import-error


from collections import deque
import json


class Link:
    """Bounded control/status topics; DDS callbacks never touch actuators."""

    def __init__(self, role, namespace="/ur12e/infer"):
        import rclpy  # pylint: disable=import-error
        from std_msgs.msg import (
            String,
        )

        rclpy.init()
        self.rclpy = rclpy
        self.message_type = String
        self.node = rclpy.create_node("ur12e_infer_" + role)
        self.messages = deque(maxlen=128)
        incoming = "control" if role == "client" else "status"
        outgoing = "status" if role == "client" else "control"
        self.publisher = self.node.create_publisher(
            String, namespace + "/" + outgoing, 10
        )
        self.subscription = self.node.create_subscription(
            String, namespace + "/" + incoming, self._receive, 10
        )

    def _receive(self, message):
        try:
            decoded = json.loads(message.data)
            if isinstance(decoded, dict):
                self.messages.append(decoded)
        except (ValueError, TypeError):
            pass

    def poll(self):
        """Drain currently available callbacks without waiting for the
        network.
        """
        self.rclpy.spin_once(self.node, timeout_sec=0)
        result = list(self.messages)
        self.messages.clear()
        return result

    def send(self, value):
        """Publish a compact status/authority message."""
        message = self.message_type()
        message.data = json.dumps(value, allow_nan=False)
        self.publisher.publish(message)

    def close(self):
        """Close ROS resources without touching device power or controller
        mode.
        """
        self.node.destroy_node()
        self.rclpy.shutdown()
