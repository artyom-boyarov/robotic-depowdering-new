from functools import reduce
from xml.etree import ElementTree as et
from xml.dom import minidom
import numpy as np
import pymeshlab as ml
import trimesh
import json
import os
import shutil

def get_obj_urdf(name, m=1.0, s=1.0):
    robot = et.Element('robot')
    robot.set('name', name)
    link = et.SubElement(robot, 'link')
    link.set('name', name)
    contact = et.SubElement(link, 'contact')
    lateral_friction = et.SubElement(contact, 'lateral_friction')
    lateral_friction.set('value', '1.0')
    rolling_friction = et.SubElement(contact, 'rolling_friction')
    rolling_friction.set('value', '1.0')
    inertia_scaling = et.SubElement(contact, 'inertia_scaling')
    inertia_scaling.set('value', '3.0')
    contact_cdm = et.SubElement(contact, 'contact_cdm')
    contact_cdm.set('value', '0.0')
    contact_erp = et.SubElement(contact, 'contact_erp')
    contact_erp.set('value', '1.0')
    inertial = et.SubElement(link, 'inertial')
    origin = et.SubElement(inertial, 'origin')
    origin.set('rpy', '0 0 0')
    origin.set('xyz', '0 0 0')
    mass = et.SubElement(inertial, 'mass')
    mass.set('value', '{}'.format(m))
    inertia = et.SubElement(inertial, 'inertia')
    inertia.set('ixx', '0')
    inertia.set('ixy', '0')
    inertia.set('ixz', '0')
    inertia.set('iyy', '0')
    inertia.set('iyz', '0')
    inertia.set('izz', '0')
    visual = et.SubElement(link, 'visual')
    origin = et.SubElement(visual, 'origin')
    origin.set('rpy', '0 0 0')
    origin.set('xyz', '0 0 0')
    geometry = et.SubElement(visual, 'geometry')
    mesh = et.SubElement(geometry, 'mesh')
    mesh.set('filename', name + '_vis.obj')
    mesh.set('scale', '{} {} {}'.format(s, s, s))
    material = et.SubElement(visual, 'material')
    material.set('name', 'blockmat')
    color = et.SubElement(material, 'color')
    color.set('rgba', '1.0 1.0 1.0 1.0')
    collision = et.SubElement(link, 'collision')
    origin = et.SubElement(collision, 'origin')
    origin.set('rpy', '0 0 0')
    origin.set('xyz', '0 0 0')
    geometry = et.SubElement(collision, 'geometry')
    mesh = et.SubElement(geometry, 'mesh')
    mesh.set('filename', name + '_col.obj')
    mesh.set('scale', '{} {} {}'.format(s, s, s))
    xml_str = minidom.parseString(et.tostring(robot)).toprettyxml(indent='  ')
    return xml_str

# Process one singular object mesh given by object_name (no extensions)
# Outputs all required files to output/
def process_obj_mesh(object_name, output):
    ms = ml.MeshSet()

    if object_name[-4:] == ".obj":
        object_name = object_name[:-4]
    rclpy.node.get_logger('mesh_processing').info(
        'Processing mesh for %s' % (object_name)
    )
    
    obj_path = os.path.join(output, object_name)
    if os.path.exists(obj_path):
        rclpy.node.get_logger('mesh_processing').info(
            'Mesh already processed; not processing again'
        )
        return
    os.makedirs(obj_path)
    rclpy.node.get_logger('mesh_processing').info(
        'Saving output meshes to %s' % (obj_path)
    )

    cad_model_dir = os.path.join(
        get_package_share_directory('robotic_depowdering'), 
        'test_parts'
    )

    ms.load_new_mesh(os.path.join(cad_model_dir, object_name + '.obj'))
    # ms.apply_filter('meshing_invert_face_orientation', forceflip=False)
    # ms.apply_filter('transform_align_to_principal_axis')
    ms.save_current_mesh(os.path.join(obj_path, object_name + '_col.obj'))
    ms.save_current_mesh(os.path.join(obj_path, object_name + '_vis.obj'))

    # These parameters below work well for the buckle. May need to be changed 
    # for other parts or dynamically chosen
    ms.apply_filter('generate_resampled_uniform_mesh', 
                    cellsize=ml.Percentage(2), 
                    mergeclosevert=True)
    ms.apply_filter('meshing_remove_connected_component_by_diameter',
                    mincomponentdiag = ml.Percentage(3)) # was 1.5

    ms.apply_filter('apply_coord_laplacian_smoothing_surface_preserving',
                    angledeg=10, iterations=10)
    ms.save_current_mesh(os.path.join(obj_path, object_name + '.obj'))
    ms.save_current_mesh(os.path.join(obj_path, object_name + '.ply'))
    xml_str = get_obj_urdf(object_name, m=1.0, s=1.0)
    with open(os.path.join(obj_path, object_name + '.urdf'), 'w') as f:
        f.write(xml_str)


def main():
    rclpy.init()
    this_node = rclpy.node.Node('mesh_processing')

    mesh_path_param_name = 'mesh_path'
    output_path_param_name = 'output_path'
    ROBOTIC_DEPOWDERING_TMP_DIR = 'tmp/'

    this_node.declare_parameter(
        mesh_path_param_name, 
        ROBOTIC_DEPOWDERING_TMP_DIR + 'meshes'
    )
    this_node.declare_parameter(
        output_path_param_name, 
        ROBOTIC_DEPOWDERING_TMP_DIR + 'meshes_output'
    )

    mesh_path = this_node.get_parameter(mesh_path_param_name)\
        .get_parameter_value().string_value
    output_path = this_node.get_parameter(output_path_param_name)\
        .get_parameter_value().string_value
    print(output_path)
    
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)
    
    process_obj_mesh(mesh_path, output_path)

if __name__ == '__main__':
    main()