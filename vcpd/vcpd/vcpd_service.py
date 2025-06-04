from robotic_depowdering_interfaces.srv import VCPDGrasp
import rclpy
from rclpy.node import Node
from .grasping.mesh_processing import process_obj_mesh
from .grasping.grasp_analysis import find_grasp

class VCPDService(Node):

    def __init__(self):
        super().__init__('vcpd_service')
        self.srv = self.create_service(VCPDGrasp, 'vcpd_get_grasp', self.get_vcpd_grasp_cb)
        
        self.declare_parameters(
            namespace='',
            parameters=[
                ('gui', False),
                ('mesh_dir', ''),
                ('verbose', False)
            ]
        )
        self.mesh_dir = self.get_parameter('mesh_dir').get_parameter_value().string_value
        self.enable_gui = self.get_parameter('gui').get_parameter_value().bool_value
        self.enable_verbose = self.get_parameter('verbose').get_parameter_value().bool_value


    def get_vcpd_grasp_cb(self, request: VCPDGrasp.Request, response: VCPDGrasp.Response):
        process_obj_mesh(request.name, self.mesh_dir)
        response = find_grasp(self.enable_gui, self.enable_verbose, request.name, self.mesh_dir)
        return response


def main():
    rclpy.init()
    vcpd_service = VCPDService()
    rclpy.spin(vcpd_service)
    rclpy.shutdown()


if __name__ == "__main__":
    main()

