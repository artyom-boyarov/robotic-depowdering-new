from robotic_depowdering_interfaces.srv import VCPDGrasp
from geometry_msgs.msg import Point
import rclpy
from rclpy.node import Node
from sys import argv

class VCPDTestClient(Node):
    def __init__(self):
        super().__init__('vcpd_test_node')
        self.client = self.create_client(VCPDGrasp, 'vcpd_get_grasp')

        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service \'vcpd_get_grasp\' not available, waiting')
        
        self.req = VCPDGrasp.Request()
    
    def send_request(self, object_name):
        self.req.name = object_name
        self.req.position = Point()
        self.req.position.x = 0.0
        self.req.position.y = 0.0
        self.req.position.z = 0.0

        return self.client.call_async(self.req)
    

def main():
    rclpy.init()

    vcpd_test_client = VCPDTestClient()

    if len(argv) < 2:
        print("Usage: ros2 run sample_vcpd_client [object_name]")
        exit(1)
    vcpd_test_client.get_logger().info("Finding a grasp for %s" % argv[1])
    future = vcpd_test_client.send_request(argv[1])
    rclpy.spin_until_future_complete(vcpd_test_client, future)

    response = future.result()
    vcpd_test_client.get_logger().info(
        "Got grasp for %s with base position at <%f, %f, %f> and width %f" %
        (argv[1], response.position.x, response.position.y, response.position.z, response.width)
    )

    vcpd_test_client.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()