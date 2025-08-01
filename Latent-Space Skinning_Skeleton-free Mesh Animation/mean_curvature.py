import torch 
import numpy as np



def extract_surface_faces(simplices):
    # simplices: (N, 4) numpy array
    # Every tetrahedron has 4 triagnle faces
    faces = []
    for tet in simplices:
        a, b, c, d = tet
        faces += [
            tuple(sorted((a, b, c))),
            tuple(sorted((a, b, d))),
            tuple(sorted((a, c, d))),
            tuple(sorted((b, c, d))),
        ]
    # Οι έδρες που εμφανίζονται μόνο 1 φορά είναι στην επιφάνεια
    from collections import Counter
    face_count = Counter(faces)
    surface_faces = [face for face, count in face_count.items() if count == 1]
    return np.array(surface_faces)



def build_laplacian_matrix(num_vertices, faces):
    """
   sparse Laplacian matrix (V x V) 
    """
    I, J = [], []
    for tri in faces:
        for i in range(3):
            vi = tri[i]
            vj = tri[(i + 1) % 3]
            I.extend([vi, vj])
            J.extend([vj, vi])

    I = torch.tensor(I, dtype=torch.long)
    J = torch.tensor(J, dtype=torch.long)
    V = torch.ones_like(I, dtype=torch.float32)
    
    A = torch.sparse_coo_tensor(torch.stack([I, J]), V, (num_vertices, num_vertices))
    D = torch.sparse.sum(A, dim=1).to_dense()
    D_inv = torch.diag(1.0 / (D + 1e-6))
    
    L = torch.eye(num_vertices) - D_inv @ A.to_dense()
    return L


def curvature_loss(predicted, target, L):
    """
    predicted, target: (V, 3) positions
    L: (V, V) Laplacian matrix
    """
    lap_pred = L @ predicted  # (V, 3)
    lap_target = L @ target
    diff = lap_pred - lap_target
    norm = torch.mean(lap_target ** 2) + 1e-8  # για αποφυγή διαίρεσης με το μηδέν
    return torch.mean(diff ** 2) / norm