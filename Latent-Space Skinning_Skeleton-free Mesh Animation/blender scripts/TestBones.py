import bpy 
import numpy as np
import mathutils
from mathutils import Matrix 

def takeBones(armature,mesh_obj):
    bones = []

    vertex_group_names = [group.name for group in mesh_obj.vertex_groups]
    for bone in armature.data.bones:
        if bone.name not in vertex_group_names:
            continue
        bones.append(bone)

    return bones

def ReadBones(filepath):
    bones_from_file = np.load(filepath)
    print(bones_from_file.shape)
    return bones_from_file

def numpy_array_to_blender_matrices(bones_matrices):
    blender_matrices = []

    for frame_matrices in bones_matrices:
        frame_blender_matrices = []
        for bone_matrix_1d in frame_matrices:
            bone_matrix_2d = bone_matrix_1d.reshape((4, 4)).tolist()
            frame_blender_matrices.append(Matrix(bone_matrix_2d))
        
        blender_matrices.append(frame_blender_matrices)

    return blender_matrices

def testMatrices(final,bones,armature_obj):

    for bone, matrix in zip(bones, final):
        bone_obj = armature_obj.pose.bones[bone.name]
        bone_obj.matrix = matrix

        # After assigning the matrix, force an update of the dependency graph
        depsgraph = bpy.context.evaluated_depsgraph_get()
        depsgraph.update()

        # Keyframe the bone transformation
        bone_obj.keyframe_insert(data_path="location")
        bone_obj.keyframe_insert(data_path="rotation_quaternion")
        bone_obj.keyframe_insert(data_path="scale")

def add_homogeneous_line(bones_matrices):
    homogeneous_line = np.array([[0, 0, 0, 1]])
    homogeneous_lines = np.tile(homogeneous_line, (bones_matrices.shape[0], bones_matrices.shape[1], 1))
    bones_matrices_with_homogeneous = np.concatenate((bones_matrices, homogeneous_lines), axis=2)
    return bones_matrices_with_homogeneous

def main():
    armature_obj_for_Test = bpy.data.objects['ArmatureTest']
    mesh_obj_walk = bpy.data.objects["SurfaceTest"]
    AnimationBones = ReadBones("boneWalking1.npy")# this return a shape of [frames,num_of_bones,matrices_of_bone]
    new = add_homogeneous_line(AnimationBones)
    print(new[0])
    new=numpy_array_to_blender_matrices(new)
   
    bones = takeBones(armature_obj_for_Test,mesh_obj_walk)
    last_frame = bpy.context.scene.frame_end
    for frame in range(0,last_frame):
        bpy.context.scene.frame_set(frame)
        testMatrices(new[frame],bones,armature_obj_for_Test)



if __name__ == "__main__":
    main()