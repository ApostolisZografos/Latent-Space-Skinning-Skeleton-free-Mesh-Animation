import numpy as np
from scipy.spatial import Delaunay
import torch

def load_mesh(file_path):
    mesh = np.load(file_path)
    print(f"Mesh dtype: {mesh.dtype}")
    
    if mesh.dtype != np.float64:
        print("Converting mesh to float64...")
        mesh = mesh.astype(np.float64)
    
    return mesh

def calculate_volume(vertices, simplices):
    # Υπολογισμός όγκου για κάθε τετράεδρο
    volume = 0.0
    for simplex in simplices:
        v0, v1, v2, v3 = [vertices[i] for i in simplex]
        v0 = np.array(v0)
        v1 = np.array(v1)
        v2 = np.array(v2)
        v3 = np.array(v3)
        
        # Δημιουργία του πίνακα για τον υπολογισμό του όγκου
        matrix = np.array([
            [v1[0] - v0[0], v1[1] - v0[1], v1[2] - v0[2]],
            [v2[0] - v0[0], v2[1] - v0[1], v2[2] - v0[2]],
            [v3[0] - v0[0], v3[1] - v0[1], v3[2] - v0[2]]
        ])
        volume += np.abs(np.linalg.det(matrix)) / 6.0
    return volume

def generate_tetrahedrons(vertices):
    # Χρησιμοποιούμε την Delaunay για να παράγουμε τα τετράεδρα από τα vertices
    delaunay = Delaunay(vertices)
    simplices = delaunay.simplices
    return simplices


def calculate_mesh_volume(mesh):
    # Υπολογίζουμε τον όγκο για κάθε mesh σε σχέση με την αρχή των αξόνων (0,0,0)
    volumes = []
    for vertices in mesh:
        vertices = np.array(vertices)
        origin = np.array([0.0, 0.0, 0.0])
        vertices_from_origin = vertices - origin
        simplices = generate_tetrahedrons(vertices_from_origin)
        

        volume = calculate_volume(vertices_from_origin, simplices)
        volumes.append(volume)
    
    return np.array(volumes)

def calculate_volume_torch(vertices, simplices, device="cuda"):
    """
    Υπολογισμός όγκου για κάθε τετράεδρο χρησιμοποιώντας PyTorch.
    :param vertices: torch.Tensor με σχήμα (N, 3), οι κορυφές του mesh.
    :param simplices: torch.Tensor με σχήμα (M, 4), τα τετράεδρα (δείκτες στις κορυφές).
    :param device: Η συσκευή (π.χ., "cuda" ή "cpu").
    :return: Συνολικός όγκος (torch.Tensor).
    """
    vertices = vertices.to(device)
    simplices = simplices.to(device)
    #print(f"Vertices shape: {vertices.shape}")
    #print(f"Simplices shape: {simplices.shape}")

    v0 = vertices[simplices[:, 0]]
    v1 = vertices[simplices[:, 1]]
    v2 = vertices[simplices[:, 2]]
    v3 = vertices[simplices[:, 3]]

    matrix = torch.stack([
        v1 - v0,
        v2 - v0,
        v3 - v0
    ], dim=1)
    det = torch.linalg.det(matrix)  # Σχήμα: (M,)
    volume = torch.abs(det) / 6.0  # Σχήμα: (M,)

    return torch.sum(volume)

def save_volume_to_npy(volume, output_path):
    np.save(output_path, np.array([volume]))


def save_tetrahedrons_for_first_frame(mesh_file, output_file):
    mesh = np.load(mesh_file) 
    first_frame = mesh[0]  # Σχήμα (7093, 3)
    simplices = generate_tetrahedrons(first_frame)
    np.save(output_file, simplices)
    print(f"Tetrahedrons for the first frame saved to {output_file}")


if __name__ == "__main__":
    file_path = "npy_files/meshWalking1.npy"
    output_path = "npy_files/mesh_volume.npy"
    
    mesh = load_mesh(file_path)
    volume = calculate_mesh_volume(mesh)
    save_volume_to_npy(volume, output_path)
    print(f"Volume saved to: {output_path}")

    save_tetrahedrons_for_first_frame(file_path, "tetrahedrons.npy")

