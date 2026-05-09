import torch
import numpy as np

def find_center_softargmax(volume, boundary, refine=True, fixed_labels=None, eps=1e-6):
    """
        3D soft-argmax 
      : volume (np.ndarray), shape: (D,H,W)
      : (z,y,x)      (   )
    """
    # Internal processing step.
    volume_tensor = torch.from_numpy(volume).float()
    volume_tensor = torch.clamp(volume_tensor, min=0)

    D, H, W = volume_tensor.shape
    volume_tensor = volume_tensor.unsqueeze(0).unsqueeze(0)  # (B=1, C=1, D, H, W)

    # Internal processing step.
    grid_z = torch.arange(D).view(1, 1, D, 1, 1).float()
    grid_y = torch.arange(H).view(1, 1, 1, H, 1).float()
    grid_x = torch.arange(W).view(1, 1, 1, 1, W).float()

    mass = volume_tensor.sum(dim=[2,3,4], keepdim=True) + eps

    com_z = (volume_tensor * grid_z).sum(dim=[2,3,4], keepdim=True) / mass
    com_y = (volume_tensor * grid_y).sum(dim=[2,3,4], keepdim=True) / mass
    com_x = (volume_tensor * grid_x).sum(dim=[2,3,4], keepdim=True) / mass

    coords = torch.cat([com_x, com_y, com_z], dim=-1)
    coords = coords.view(3).cpu().numpy()  # numpy float array (x, y, z)
    
    if not refine:
        return tuple(coords)  
    
    # Internal processing step.
    return refinement(coords, boundary, fixed_labels)

# ==========================================
# Internal processing step.
# ==========================================

def get_surface_cloud(boundary):
    """  boundary map        """
    boundary_np = boundary.cpu().numpy() if isinstance(boundary, torch.Tensor) else boundary
    boundary_np = np.squeeze(boundary_np)
    max_value = boundary_np.max()
    
    # Internal processing step.
    surface_mask = boundary_np == max_value
    if not np.any(surface_mask):
        return None
    
    # Internal processing step.
    min_indices = np.argwhere(surface_mask)
    surface_pts = np.stack([min_indices[:, 2], min_indices[:, 1], min_indices[:, 0]], axis=1)
    return surface_pts

def project_to_surface(coord, surface_pts):
    """              """
    if surface_pts is None:
        return coord
    
    coord_arr = np.array(coord)
    distances = np.linalg.norm(surface_pts - coord_arr, axis=1)
    closest_idx = np.argmin(distances)
    return tuple(surface_pts[closest_idx])

def fit_plane_constrained_by_line(p1, p2, other_points):
    """
                   p1-p2     other_points (Logic 2)
    """
    p1 = np.array(p1)
    p2 = np.array(p2)
    pts = np.array(other_points)
    
    # Internal processing step.
    u = p2 - p1
    norm_u = np.linalg.norm(u)
    if norm_u < 1e-6:
        return pts
    u = u / norm_u
    
    # Internal processing step.
    if np.abs(u[0]) > 0.9:
        tmp = np.array([0, 1, 0])
    else:
        tmp = np.array([1, 0, 0])
    
    v = tmp - np.dot(tmp, u) * u
    v = v / np.linalg.norm(v)
    w = np.cross(u, v)
    
    # Internal processing step.
    rel_pts = pts - p1
    proj_v = np.dot(rel_pts, v)
    proj_w = np.dot(rel_pts, w)
    
    # Internal processing step.
    X_2d = np.column_stack((proj_v, proj_w))
    U, S, Vh = np.linalg.svd(X_2d)
    normal_2d = Vh[-1] 
    
    # Internal processing step.
    plane_normal = normal_2d[0] * v + normal_2d[1] * w
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    
    # Internal processing step.
    dists = np.dot(pts - p1, plane_normal)
    projected_points = pts - np.outer(dists, plane_normal)
    
    return projected_points

def project_points_to_best_fit_plane(points):
    """
    Logic 3:                 
    """
    pts = np.array(points)
    
    # Internal processing step.
    center = np.mean(pts, axis=0)
    centered_pts = pts - center
    
    # Internal processing step.
    # Internal processing step.
    U, S, Vh = np.linalg.svd(centered_pts)
    normal = Vh[-1] 
    
    # Internal processing step.
    # P_proj = P - dot(P - center, normal) * normal
    dists = np.dot(centered_pts, normal) # (N,)
    projected_pts = pts - np.outer(dists, normal)
    
    return projected_pts

def refine_group_geometry(coords_dict, surface_pts):
    """
            
    """
    final_coords = coords_dict.copy()
    
    # Internal processing step.
    # Internal processing step.
    direct_group = ["ss", "sm", "si", "AA", "AP", "C", "bi2", "bi12"]
    
    for lbl in direct_group:
        if lbl in final_coords:
            final_coords[lbl] = project_to_surface(final_coords[lbl], surface_pts)
            
    # Internal processing step.
    plane_group = ["sm", "si", "gc", "SCP6", "SCP12"]
    if all(lbl in final_coords for lbl in plane_group):
        # Internal processing step.
        p_sm = final_coords["sm"]
        p_si = final_coords["si"]
        
        others = [final_coords["gc"], final_coords["SCP6"], final_coords["SCP12"]]
        projected_others = fit_plane_constrained_by_line(p_sm, p_si, others)
        
        # Internal processing step.
        final_coords["gc"] = project_to_surface(projected_others[0], surface_pts)
        final_coords["SCP6"] = project_to_surface(projected_others[1], surface_pts)
        final_coords["SCP12"] = project_to_surface(projected_others[2], surface_pts)

    # Internal processing step.
    bi_subset_labels = ["bi3", "bi4", "bi5", "bi6", "bi7", "bi8", "bi9"]
    present_bi_labels = [lbl for lbl in bi_subset_labels if lbl in final_coords]
    
    # Internal processing step.
    if len(present_bi_labels) >= 3:
        # Internal processing step.
        pts = [final_coords[lbl] for lbl in present_bi_labels]
        
        # Internal processing step.
        planar_pts = project_points_to_best_fit_plane(pts)
        
        # Internal processing step.
        for i, lbl in enumerate(present_bi_labels):
            final_coords[lbl] = project_to_surface(planar_pts[i], surface_pts)
    else:
        # Internal processing step.
        for lbl in present_bi_labels:
            final_coords[lbl] = project_to_surface(final_coords[lbl], surface_pts)

    return final_coords

# ==========================================
# Internal processing step.
# ==========================================

def refinement(coords, boundary, fixed_labels=None):
    """
    Refinement     
      :
        coords: dict {label: (x,y,z)}   tuple (x,y,z)
        boundary:    
    """
    surface_pts = get_surface_cloud(boundary)
    if surface_pts is None:
        return coords

    if isinstance(coords, dict):
        return refine_group_geometry(coords, surface_pts)
    else:
        # Internal processing step.
        return project_to_surface(coords, surface_pts)
