#include "rdr/accel.h"

#include "rdr/canary.h"
#include "rdr/interaction.h"
#include "rdr/math_aliases.h"
#include "rdr/platform.h"
#include "rdr/shape.h"

RDR_NAMESPACE_BEGIN

/* ===================================================================== *
 *
 * AABB Implementations
 *
 * ===================================================================== */

bool AABB::isOverlap(const AABB &other) const {
  return ((other.low_bnd[0] >= this->low_bnd[0] &&
           other.low_bnd[0] <= this->upper_bnd[0]) ||
          (this->low_bnd[0] >= other.low_bnd[0] &&
           this->low_bnd[0] <= other.upper_bnd[0])) &&
         ((other.low_bnd[1] >= this->low_bnd[1] &&
           other.low_bnd[1] <= this->upper_bnd[1]) ||
          (this->low_bnd[1] >= other.low_bnd[1] &&
           this->low_bnd[1] <= other.upper_bnd[1])) &&
         ((other.low_bnd[2] >= this->low_bnd[2] &&
           other.low_bnd[2] <= this->upper_bnd[2]) ||
          (this->low_bnd[2] >= other.low_bnd[2] &&
           this->low_bnd[2] <= other.upper_bnd[2]));
}

bool AABB::intersect(const Ray &ray, Float *t_in, Float *t_out) const {
  // TODO(HW3): implement ray intersection with AABB.
  // ray distance for two intersection points are returned by pointers.
  //
  // This method should modify t_in and t_out as the "time"
  // when the ray enters and exits the AABB respectively.
  //
  // And return true if there is an intersection, false otherwise.
  //
  // Useful Functions:
  // @see Ray::safe_inverse_direction
  //    for getting the inverse direction of the ray.
  // @see Min/Max/ReduceMin/ReduceMax
  //    for vector min/max operations.
  // Get the inverse direction for efficiency
  Vec3f inv_dir;
  for (int i = 0; i < 3; i++) {
    inv_dir[i] =
        (std::abs(ray.direction[i]) < 1e-8f) ? 1e8f : 1.0f / ray.direction[i];
  }

  // Calculate intersection with each pair of slabs
  Vec3f t0 = (low_bnd - ray.origin) * inv_dir;
  Vec3f t1 = (upper_bnd - ray.origin) * inv_dir;

  // Make sure t0 is the near intersection and t1 is the far intersection
  Vec3f t_near = Min(t0, t1);
  Vec3f t_far = Max(t0, t1);

  // Find the largest t_near and smallest t_far
  Float t_enter = std::max({t_near[0], t_near[1], t_near[2]});
  Float t_exit = std::min({t_far[0], t_far[1], t_far[2]});

  // Check if there is an intersection
  if (t_enter > t_exit || t_exit < 0.0f) {
    return false;
  }

  // Set the output parameters
  *t_in = t_enter;
  *t_out = t_exit;

  return true;
}

/* ===================================================================== *
 *
 * Accelerator Implementations
 *
 * ===================================================================== */

bool TriangleIntersect(Ray &ray, const uint32_t &triangle_index,
                       const ref<TriangleMeshResource> &mesh,
                       SurfaceInteraction &interaction) {
  using InternalScalarType = Double;
  using InternalVecType = Vec<InternalScalarType, 3>;

  AssertAllValid(ray.direction, ray.origin);
  AssertAllNormalized(ray.direction);

  const auto &vertices = mesh->vertices;
  const Vec3u v_idx(&mesh->v_indices[3 * triangle_index]);
  assert(v_idx.x < mesh->vertices.size());
  assert(v_idx.y < mesh->vertices.size());
  assert(v_idx.z < mesh->vertices.size());

  InternalVecType dir = Cast<InternalScalarType>(ray.direction);
  InternalVecType v0 = Cast<InternalScalarType>(vertices[v_idx[0]]);
  InternalVecType v1 = Cast<InternalScalarType>(vertices[v_idx[1]]);
  InternalVecType v2 = Cast<InternalScalarType>(vertices[v_idx[2]]);

  // TODO(HW3): implement ray-triangle intersection test.
  // You should compute the u, v, t as InternalScalarType
  //
  //   InternalScalarType u = ...;
  //   InternalScalarType v = ...;
  //   InternalScalarType t = ...;
  //
  // And exit early with `return false` if there is no intersection.
  //
  // The intersection points is denoted as:
  // (1 - u - v) * v0 + u * v1 + v * v2 == ray.origin + t * ray.direction
  // where the left side is the barycentric interpolation of the triangle
  // vertices, and the right side is the parametric equation of the ray.
  //
  // You should also make sure that:
  // u >= 0, v >= 0, u + v <= 1, and, ray.t_min <= t <= ray.t_max
  //
  // Useful Functions:
  // You can use @see Cross and @see Dot for determinant calculations.

  // Möller-Trumbore intersection algorithm
  InternalVecType edge1 = v1 - v0;
  InternalVecType edge2 = v2 - v0;
  InternalVecType h = Cross(dir, edge2);
  InternalScalarType a = Dot(edge1, h);

  // Ray is parallel to triangle
  if (std::abs(a) < 1e-10) {
    return false;
  }

  InternalScalarType f = InternalScalarType(1.0) / a;
  InternalVecType s = Cast<InternalScalarType>(ray.origin) - v0;
  InternalScalarType u = f * Dot(s, h);

  // Check if u is in valid range [0, 1]
  if (u < 0.0 || u > 1.0) {
    return false;
  }

  InternalVecType q = Cross(s, edge1);
  InternalScalarType v = f * Dot(dir, q);

  // Check if v is in valid range and u + v <= 1
  if (v < 0.0 || u + v > 1.0) {
    return false;
  }

  // Calculate t
  InternalScalarType t = f * Dot(edge2, q);

  // Check if t is in valid range [t_min, t_max]
  if (t < ray.t_min || t > ray.t_max) {
    return false;
  }

  // We will reach here if there is an intersection

  CalculateTriangleDifferentials(interaction,
                                 {static_cast<Float>(1 - u - v),
                                  static_cast<Float>(u), static_cast<Float>(v)},
                                 mesh, triangle_index);
  AssertNear(interaction.p, ray(t));
  assert(ray.withinTimeRange(t));
  ray.setTimeMax(t);
  return true;
}

void Accel::setTriangleMesh(const ref<TriangleMeshResource> &mesh) {
  // Build the bounding box
  AABB bound(Vec3f(Float_INF, Float_INF, Float_INF),
             Vec3f(Float_MINUS_INF, Float_MINUS_INF, Float_MINUS_INF));
  for (auto &vertex : mesh->vertices) {
    bound.low_bnd = Min(bound.low_bnd, vertex);
    bound.upper_bnd = Max(bound.upper_bnd, vertex);
  }

  this->mesh = mesh;   // set the pointer
  this->bound = bound; // set the bounding box
}

void Accel::build() {}

AABB Accel::getBound() const { return bound; }

bool Accel::intersect(Ray &ray, SurfaceInteraction &interaction) const {
  bool success = false;
  for (int i = 0; i < mesh->v_indices.size() / 3; i++)
    success |= TriangleIntersect(ray, i, mesh, interaction);
  return success;
}

RDR_NAMESPACE_END
