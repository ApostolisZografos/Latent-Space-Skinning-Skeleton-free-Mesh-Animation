import bpy 
import numpy as np
#---------------bones ------------------>
def takeBones(armature,mesh_obj):
    bones = []

    vertex_group_names = [group.name for group in mesh_obj.vertex_groups]
    for bone in armature.data.bones:
        if bone.name not in vertex_group_names:
            continue
        bones.append(bone)

    return bones

def findLocalMatrices(bones):
    local_matrices=[]
    for bone in bones:
        bone_matrix_local = bone.matrix_local
        local_matrices.append(bone_matrix_local)
    return local_matrices

def matrix_world(armature_ob, bone_name):
    local = armature_ob.data.bones[bone_name].matrix_local
    basis = armature_ob.pose.bones[bone_name].matrix_basis

    parent = armature_ob.pose.bones[bone_name].parent
    #print(" this is the parent -> ",parent)
    if parent == None:
        return  local @ basis
    else:
        parent_local = armature_ob.data.bones[parent.name].matrix_local
        return matrix_world(armature_ob, parent.name) @ (parent_local.inverted() @ local) @ basis

def calculate_bones_matrices(bones,armature):
    final_matrices = []
    armature_matrix_world = armature.matrix_world
    for bone in bones:
        bone_matrix_world = armature_matrix_world @ matrix_world(armature,bone.name)
        final_matrices.append(bone_matrix_world)

    return final_matrices

def crop_matrix(matrix_list):
    cropped = []
    for matrix in matrix_list:
        matrix_np = np.array(matrix)
        cropped_matrix = matrix_np[:3, :4]
        cropped.append(cropped_matrix)
    return cropped

def find_matrices_Frames(armature,bones,last_frame):
    start_frame = 0 
    end_frame = last_frame
    final_matrices_full = []
    cropped_matrices_full = [] 

    for frame in range(start_frame, end_frame +1):
        bpy.context.scene.frame_set(frame)
        final_matrices_frame = calculate_bones_matrices(bones,armature)
        final_matrices_full.append(final_matrices_frame)

        final_matrices_frame_cropped = crop_matrix(final_matrices_frame)
        cropped_matrices_full.append(final_matrices_frame_cropped)
        
    # return the full matrices and the cropped matrices 
    return final_matrices_full,cropped_matrices_full

#----------------mesh------------------------------->
def fastWayCaclulateMesh(final_matrices,mesh_obj,bones):
    
    local_matrices= findLocalMatrices(bones)
    # print(len(local_matrices))
    mesh = mesh_obj.data
    vertices_co = []
    final_matrices = np.array(final_matrices)
    # print(len(final_matrices))
    vertices_co = np.hstack((np.array([v.co for v in mesh.vertices]), np.ones((len(mesh.vertices), 1))))
    transformed_coordinates = np.zeros((len(mesh.vertices), 3))

    inv_local_matrices = np.linalg.inv(local_matrices)

    for vertex in mesh.vertices:
        weights = np.array([group.weight for group in vertex.groups])
        # print(weights.shape)
        bone_indices = [group.group for group in vertex.groups]
        # print(len(bone_indices))
        v_relative_homogeneous_all = vertices_co[vertex.index] @ inv_local_matrices[bone_indices].transpose((0, 2, 1))

        bone_matrices = final_matrices[bone_indices]

        transformed_vertices = (bone_matrices @ v_relative_homogeneous_all[..., None]).squeeze()

        transformed_vertex = (transformed_vertices * weights[..., None]).sum(axis=0)[:3]

        transformed_coordinates[vertex.index] = transformed_vertex


    vertices_array = np.array(transformed_coordinates)
    return vertices_array

def mesh_for_all_frames(final_full,last_frame,mesh_obj,bones):
    print("calculate mesh vertices")
    mesh_for_all_frames = [] 
    for frame in range(0,last_frame+1):
        mesh_for_all_frames.append(fastWayCaclulateMesh(final_full[frame],mesh_obj,bones))
    return mesh_for_all_frames
#----------------------------------------------------->

def dot_visualizeTheMeshVertices(vertices_array):
    # Ensure your context is correct, bpy.context.area.type == 'VIEW_3D'
    
    # Iterate through each frame's vertices
    for frame_idx, frame_vertices in enumerate(vertices_array):
        # Create a new mesh and object for each frame
        mesh = bpy.data.meshes.new(f"Frame_{frame_idx}_Mesh")
        obj = bpy.data.objects.new(f"Frame_{frame_idx}", mesh)
        
        # Link the object to the scene
        bpy.context.collection.objects.link(obj)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        
        # Create mesh from_pydata
        mesh.from_pydata(frame_vertices, [], [])
        mesh.update()

        # Deselect the object and move to the next frame
        obj.select_set(False)

    # Re-select the first frame object and make it active
    first_obj = bpy.data.objects.get(f"Frame_0")
    if first_obj:
        first_obj.select_set(True)
        bpy.context.view_layer.objects.active = first_obj

#This will make flatten frames to use it in the txt BoneForFrames.txt {it convert the final_full to np.array final full}
def makeVectorsInsteadOFMatrices(cropped_final_full):
    flattened_frames = []
    for frame in cropped_final_full:
        flattened_bone_matrices = []
        for bone_matrix in frame:
            flattened_matrix = np.array(bone_matrix).flatten()
            flattened_bone_matrices.append(flattened_matrix)
        flattened_frames.append(flattened_bone_matrices)
    return flattened_frames

def CreateBonesFile(cropped_final_full,filePath):
    FinalMatrices_nparray = makeVectorsInsteadOFMatrices(cropped_final_full)
    np.save(filePath,FinalMatrices_nparray)

def makeMeshVertices(mesh_for_all_frames,filePath):
    print("Making np file for mesh vertices of hole animation (for all frames)")
    all_mesh = np.array(mesh_for_all_frames)
    np.save(filePath, all_mesh)

def makeTheData(armature,mesh_obj,filepath1,filepath2):
    armature = bpy.data.objects[armature]
    mesh_obj = bpy.data.objects[mesh_obj]
    last_frame = bpy.context.scene.frame_end
    bones_test = takeBones(armature,mesh_obj)
    final_full, cropped_final_full = find_matrices_Frames(armature,bones_test,last_frame)
    CreateBonesFile(cropped_final_full,filepath1)
    mesh_full= mesh_for_all_frames(final_full,last_frame,mesh_obj,bones_test) 
    makeMeshVertices(mesh_full,filepath2)

    #visualization with dots.
    mesh=np.array(mesh_full)
    dot_visualizeTheMeshVertices(mesh)
    
def get_output_path(filename):
    return r"npy_files" + "/" + filename


def main():
    #makeTheData("ArmatureWalkingTest", "MeshMyWalkingTest", get_output_path("boneMyWalkingTest.npy"),get_output_path("meshMyWalkingTest.npy"))
    #makeTheData("ArmatureJumpTest", "MeshJumpTest", get_output_path("boneMyJumpTest.npy"),get_output_path("meshMyJumpTest.npy"))
    '''
    makeTheData("Armature","human","Jumpbones.npy","JumpMesh.npy")
    makeTheData("Armature.001","human.001","Jumpbones1.npy","JumpMesh1.npy")
    makeTheData("Armature.002","human.002","Jumpbones2.npy","JumpMesh2.npy")
    makeTheData("Armature.003","human.003","Jumpbones3.npy","JumpMesh3.npy")
    makeTheData("Armature.004","human.004","Jumpbones4.npy","JumpMesh4.npy")

    print("Generate head Animations")
    makeTheData("Rotate_head_1","humanRotate1","rotate1bones.npy","rotate1Mesh.npy")
    makeTheData("Rotate_head_2","humanRotate2","rotate2bones.npy","rotate2Mesh.npy")   
    makeTheData("Rotate_head_3","humanRotate3","rotate3bones.npy","rotate3Mesh.npy")   
    makeTheData("Rotate_head_4","humanRotate4","rotate4bones.npy","rotate4Mesh.npy")
    makeTheData("Rotate_head_5","humanRotate5","rotate5bones.npy","rotate5Mesh.npy")
    makeTheData("Rotate_head_6","humanRotate6","rotate6bones.npy","rotate6Mesh.npy")

    print("Generate Walking animations")
    makeTheData("ArmatureWalking1","humanWalking1","boneWalking1.npy","meshWalking1.npy")
    makeTheData("ArmatureWalking2","humanWalking2","boneWalking2.npy","meshWalking2.npy")
    makeTheData("ArmatureWalking3","humanWalking3","boneWalking3.npy","meshWalking3.npy")
    makeTheData("ArmatureWalking4","humanWalking4","boneWalking4.npy","meshWalking4.npy")
    makeTheData("ArmatureWalking5","humanWalking5","boneWalking5.npy","meshWalking5.npy")
    makeTheData("ArmatureWalking6","humanWalking6","boneWalking6.npy","meshWalking6.npy")

    print("Generate pointing animations")
    makeTheData("pointing1","pointing1Mesh","bonePointing1.npy","meshPointing1.npy")
    makeTheData("pointing2","pointing2Mesh","bonePointing2.npy","meshPointing2.npy")
    makeTheData("pointing3","pointing3Mesh","bonePointing3.npy","meshPointing3.npy")
    makeTheData("pointing4","pointing4Mesh","bonePointing4.npy","meshPointing4.npy")

    print("Generate turning animations")
    makeTheData("turn1","turn1Mesh","boneTurn1.npy","meshTurn1.npy")
    makeTheData("turn2","turn2Mesh","boneTurn2.npy","meshTurn2.npy")
    makeTheData("turn3","turn3Mesh","boneTurn3.npy","meshTurn3.npy")
    makeTheData("turn4","turn4Mesh","boneTurn4.npy","meshTurn4.npy")
    makeTheData("turn5","turn5Mesh","boneTurn5.npy","meshTurn5.npy")
    makeTheData("turn6","turn6Mesh","boneTurn6.npy","meshTurn6.npy")
    makeTheData("turn7","turn7Mesh","boneTurn7.npy","meshTurn7.npy")
    makeTheData("turn8","turn8Mesh","boneTurn8.npy","meshTurn8.npy")

    print("Generate bot animations jump")
    makeTheData("botjumpbones1","BotJump1","botjumpbones1.npy","botjumpmesh1.npy")
    makeTheData("botjumpbones2","BotJump2","botjumpbones2.npy","botjumpmesh2.npy")
    makeTheData("botjumpbones3","BotJump3","botjumpbones3.npy","botjumpmesh3.npy")
    makeTheData("botjumpbones4","BotJump4","botjumpbones4.npy","botjumpmesh4.npy")
    '''


if __name__ == "__main__":
    main()
