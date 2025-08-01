# run this code to find if the model will generete properly the mesh after the training seaction
# For evaluation 
# this code works properly if the model script after training gives an npy file after training with the script newModelWithForceTeaching.py
import bpy 
import numpy as np
import mathutils
from mathutils import Matrix
import time 

# visualize with dots the mesh , not using
def dot_visualizeTheMeshVerices(vertices_array):
    # Create a new mesh object to visualize the transformed vertices
    new_mesh = bpy.data.meshes.new("TransformedMesh")
    new_mesh.from_pydata(vertices_array, [], [])

    # Create a new object and link it to the scene
    new_obj = bpy.data.objects.new("TransformedObject", new_mesh)
    bpy.context.collection.objects.link(new_obj)

    # Set the new object as the active object and select it
    bpy.context.view_layer.objects.active = new_obj
    new_obj.select_set(True)

    # Switch to object mode and update the view
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.select_all(action='DESELECT')
    new_obj.select_set(True)
    bpy.context.view_layer.objects.active = new_obj
# visualize the rest pose of the character and using for reset 
def call_rest_pose(mesh_object,mesh_target):
    bpy.context.view_layer.objects.active = mesh_object
    bpy.ops.object.mode_set(mode='EDIT')

    bpy.ops.mesh.select_all(action='SELECT')
    bpy.ops.mesh.quads_convert_to_tris()

    
    for vertex , vertex_rest in zip(mesh_object.data.vertices , mesh_target.data.vertices):
        vertex_rest.co = vertex.co

    mesh_target.data.update()
    
    bpy.ops.object.mode_set(mode='OBJECT')

def visualize_mesh(predicted_mesh,mesh_obj):
    mesh = mesh_obj.data
    print(predicted_mesh)
    for i, vertex in enumerate(mesh.vertices):
        vertex.co = predicted_mesh[i]
    mesh.update()
    
def visualize_vertices_RestPose(filepath,mesh_obj_test):
    all_vertices = np.load(filepath)
    vertices = len(all_vertices)
    reshaped_array_verices  = all_vertices.reshape(vertices,3)
    visualize_mesh(reshaped_array_verices,mesh_obj_test)

def visualize_vertices(filepath,mesh_obj_test,visualized_frame):
    all_vertices = np.load(filepath)
    vertices = len(all_vertices[1])
    reshaped_array_verices  = all_vertices.reshape(num_frames,vertices,3)
    
    #dot_visualizeTheMeshVerices(reshaped_array_verices[41])
    visualize_mesh(reshaped_array_verices[visualized_frame],mesh_obj_test)

def visualize_vertices_fromModel(filepath,mesh_obj_test,visualized_frame):
    all_vertices = np.load(filepath)
    print(all_vertices.shape)
    num_frames=all_vertices.shape[0]
    vertices = len(all_vertices[1])//3 
    reshaped_array_verices  = all_vertices.reshape(num_frames,vertices,3) 
    visualize_mesh(reshaped_array_verices[visualized_frame],mesh_obj_test)

#this has loop for visualize the animation 
def visualize_vertices_motionblending(filepath,mesh_obj_test):
    time_delay = 0.5
    all_vertices = np.load(filepath)
    num_frames=all_vertices.shape[0]
    vertices = all_vertices.shape[1]//3 
    reshaped_array_verices  = all_vertices.reshape(num_frames,vertices,3)
    for frame in range(num_frames):
        visualize_mesh(reshaped_array_verices[frame],mesh_obj_test)
        bpy.ops.wm.redraw_timer(type='DRAW_WIN_SWAP', iterations=1) # for sceen update
        time.sleep(time_delay)


# mesh_obj = bpy.data.objects["Alpha_Surface"]
mesh_obj_test = bpy.data.objects["Surface"]
num_frames =  bpy.context.scene.frame_end +1 
visualized_frame = 1

visualize_vertices_motionblending("npy_files/Synthesis1.npy",mesh_obj_test)
#visualize_vertices_fromModel("UnseenDataResults.npy",mesh_obj_test,visualized_frame)
# visualize_vertices("MeshJoy.npy",mesh_obj_test,visualized_frame)
# visualize_vertices("MeshJumping.npy",mesh_obj_test,visualized_frame)
# visualize_vertices("MeshRifle.npy",mesh_obj_test,visualized_frame)
# visualize_vertices("MeshWalking.npy",mesh_obj_test,visualized_frame)
#visualize_vertices_RestPose("RestPose.npy",mesh_obj_test)
