import numpy as np
import jax
from jax import grad
import jax.numpy as jnp


def mesh_to_polyscope(mesh, wrap_x=True, wrap_y=True):
    n, m, _ = mesh.shape

    n_faces = n if wrap_x else n - 1
    m_faces = m if wrap_y else m - 1

    ii, jj = np.meshgrid(np.arange(n), np.arange(m))
    ii = ii.T
    jj = jj.T
    coords = jj + m * ii

    faces = np.zeros((n_faces, m_faces, 4), int)
    for i in range(n_faces):
        for j in range(m_faces):
            faces[i, j, 0] = coords[i, j]
            faces[i, j, 1] = coords[(i + 1) % n, j]
            faces[i, j, 2] = coords[(i + 1) % n, (j + 1) % m]
            faces[i, j, 3] = coords[i, (j + 1) % m]
            # faces[i, j, 0] = j + (i * n)
            # faces[i, j, 1] = ((j + 1) % m) + (i * n)
            # faces[i, j, 2] = ((j + 1) % m) + ((i + 1) % n) * n
            # faces[i, j, 3] = j + ((i + 1) % n) * n

    mesh_ = mesh.reshape(-1, 3)
    faces_ = faces.reshape(-1, 4)

    return mesh_, faces_


def mesh_to_polyscope_triangular(mesh, wrap_x=True, wrap_y=True):
    n, m, _ = mesh.shape

    n_faces = n if wrap_x else n - 1
    m_faces = m if wrap_y else m - 1

    ii, jj = np.meshgrid(np.arange(n), np.arange(m))
    ii = ii.T
    jj = jj.T
    coords = jj + m * ii

    faces = np.zeros((n_faces, m_faces * 2, 3), int)
    for i in range(n_faces):
        for j in range(m_faces):
            faces[i, 2 * j, 0] = coords[i, j]
            faces[i, 2 * j, 1] = coords[(i + 1) % n, j]
            faces[i, 2 * j, 2] = coords[i, (j + 1) % m]

            faces[i, (2 * j) + 1, 0] = coords[(i + 1) % n, j]
            faces[i, (2 * j) + 1, 1] = coords[(i + 1) % n, (j + 1) % m]
            faces[i, (2 * j) + 1, 2] = coords[i, (j + 1) % m]

    mesh_ = mesh.reshape(-1, 3)
    faces_ = faces.reshape(-1, 3)

    return mesh_, faces_


def project_to_3d(M, V, m_to_3d, projection_matrix_to_3d):
    X = m_to_3d(M)
    Y = (
        jnp.swapaxes(projection_matrix_to_3d(M), -1, -2) @ V[..., np.newaxis]
    ).squeeze()
    return X, Y


def cylinder_m_to_3d(M):
    theta, x = M[..., 0], M[..., 1]
    # np.take(M, 0, -1), np.take(M, 1, -1)
    s = jnp.sin(theta)
    c = jnp.cos(theta)
    return jnp.stack([s, c, x], axis=-1)


def cylinder_projection_matrix_to_3d_1(M):
    theta, x = M[..., 0], M[..., 1]
    # np.take(M, 0, -1), np.take(M, 1, -1)
    s = jnp.sin(theta)
    c = jnp.cos(theta)
    z = jnp.zeros_like(c)
    e1 = jnp.stack([c, -s, z], axis=-1)
    e2 = jnp.zeros_like(e1)
    return jnp.stack(
        [
            e1,
            # e2
        ],
        axis=-2,
    )


def cylinder_projection_matrix_to_3d(M):
    theta, x = M[..., 0], M[..., 1]
    # np.take(M, 0, -1), np.take(M, 1, -1)
    s = jnp.sin(theta)
    c = jnp.cos(theta)
    z = jnp.zeros_like(c)
    o = jnp.ones_like(c)
    e1 = jnp.stack([c, -s, z], axis=-1)
    e2 = jnp.stack([z, z, o], axis=-1)
    return jnp.stack(
        [
            e1,
            e2,
        ],
        axis=-2,
    )


def t2_m_to_3d(M, R=3, r=1):
    theta1, theta2 = jnp.take(M, 0, -1), jnp.take(M, 1, -1)
    s1 = jnp.sin(theta1)
    c1 = jnp.cos(theta1)
    s2 = jnp.sin(theta2)
    c2 = jnp.cos(theta2)
    return jnp.stack([(R + r * c1) * c2, (R + r * c1) * s2, r * s1], axis=-1)


def t2_projection_matrix_to_3d(M, R=3, r=1):
    # # theta1, theta2 = M[...,0], M[...,1]
    # theta1, theta2 = jnp.take(M, 0, -1), jnp.take(M, 1, -1)
    # s1 = jnp.sin(theta1)
    # c1 = jnp.cos(theta1)
    # s2 = jnp.sin(theta2)
    # c2 = jnp.cos(theta2)
    # z = jnp.zeros_like(theta1)
    # e1 = jnp.stack([-r * s1 * c2, -r * s1 * s2, r * c1], axis=-1)
    # e2 = jnp.stack([-s2, z, z], axis=-1)
    # return jnp.stack(
    #     [
    #         e1,
    #         e2,
    #     ],
    #     axis=-2,
    # )
    grad_proj = jnp.stack(
        [jax.vmap(grad(lambda m: t2_m_to_3d(m)[..., i]))(M) for i in range(3)], axis=-1
    )
    return grad_proj / jnp.linalg.norm(grad_proj, axis=-1)[..., np.newaxis]
