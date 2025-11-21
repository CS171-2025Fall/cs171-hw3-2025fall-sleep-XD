#include "rdr/integrator.h"

#include <omp.h>

#include "rdr/bsdf.h"
#include "rdr/camera.h"
#include "rdr/canary.h"
#include "rdr/film.h"
#include "rdr/halton.h"
#include "rdr/interaction.h"
#include "rdr/light.h"
#include "rdr/math_aliases.h"
#include "rdr/math_utils.h"
#include "rdr/platform.h"
#include "rdr/properties.h"
#include "rdr/ray.h"
#include "rdr/scene.h"
#include "rdr/sdtree.h"

RDR_NAMESPACE_BEGIN

/* ===================================================================== *
 *
 * Intersection Test Integrator's Implementation
 *
 * ===================================================================== */

void IntersectionTestIntegrator::render(ref<Camera> camera, ref<Scene> scene) {
  // Statistics
  std::atomic<int> cnt = 0;

  const Vec2i &resolution = camera->getFilm()->getResolution();
#pragma omp parallel for schedule(dynamic)
  for (int dx = 0; dx < resolution.x; dx++) {
    ++cnt;
    if (cnt % (resolution.x / 10) == 0)
      Info_("Rendering: {:.02f}%", cnt * 100.0 / resolution.x);
    Sampler sampler;
    for (int dy = 0; dy < resolution.y; dy++) {
      sampler.setPixelIndex2D(Vec2i(dx, dy));
      for (int sample = 0; sample < spp; sample++) {
        // TODO(HW3): generate #spp rays for each pixel and use Monte Carlo
        // integration to compute radiance.
        //
        // Useful Functions:
        //
        // @see Sampler::getPixelSample for getting the current pixel sample
        // as Vec2f.
        //
        // @see Camera::generateDifferentialRay for generating rays given
        // pixel sample positions as 2 floats.

        // You should assign the following two variables
        // const Vec2f &pixel_sample = ...
        // auto ray = ...

        // After you assign pixel_sample and ray, you can uncomment the
        // following lines to accumulate the radiance to the film.
        //
        //
        // Accumulate radiance
        // assert(pixel_sample.x >= dx && pixel_sample.x <= dx + 1);
        // assert(pixel_sample.y >= dy && pixel_sample.y <= dy + 1);
        // const Vec3f &L = Li(scene, ray, sampler);
        // camera->getFilm()->commitSample(pixel_sample, L);
        const Vec2f &pixel_sample = sampler.getPixelSample();

        // 根据采样位置生成光线
        auto ray =
            camera->generateDifferentialRay(pixel_sample.x, pixel_sample.y);

        assert(pixel_sample.x >= dx && pixel_sample.x <= dx + 1);
        assert(pixel_sample.y >= dy && pixel_sample.y <= dy + 1);
        const Vec3f &L = Li(scene, ray, sampler);
        camera->getFilm()->commitSample(pixel_sample, L);
      }
    }
  }
}

Vec3f IntersectionTestIntegrator::Li(ref<Scene> scene, DifferentialRay &ray,
                                     Sampler &sampler) const {
  Vec3f color(0.0);
  Vec3f throughput(1.0); // 累积光通量

  // 记录是否找到了非透明的漫反射表面
  bool diffuse_found = false;
  SurfaceInteraction interaction;

  for (int i = 0; i < max_depth; ++i) {
    //
    interaction = SurfaceInteraction();
    bool intersected = scene->intersect(ray, interaction);

    if (!intersected) {
      break;
    }

    // 判断材质类型
    bool is_ideal_diffuse =
        dynamic_cast<const IdealDiffusion *>(interaction.bsdf) != nullptr;
    bool is_perfect_refraction =
        dynamic_cast<const PerfectRefraction *>(interaction.bsdf) != nullptr;

    // 设置出射方向
    interaction.wo = -ray.direction;

    if (is_perfect_refraction) {
      Vec3f wi;
      Float pdf;
      Vec3f f = interaction.bsdf->sample(interaction, sampler, &pdf);

      throughput *= f;

      ray = interaction.spawnRay(interaction.wi);

      continue;
    }

    if (is_ideal_diffuse) {
      diffuse_found = true;
      break;
    }

    break;
  }

  if (diffuse_found) {
    color = throughput * directLighting(scene, interaction, sampler);
  }

  return color;
}

Vec3f IntersectionTestIntegrator::directLighting(
    ref<Scene> scene, SurfaceInteraction &interaction, Sampler &sampler) const {

  Vec3f total_color(0.0f);

  // 定义局部结构体
  struct ManualAreaLight {
    Vec3f center;
    Vec3f flux;
    Vec2f size;
    Vec3f normal;
  };

  std::vector<ManualAreaLight> lights;

  lights.push_back({
      point_light_position, point_light_flux, Vec2f(0.5f, 0.5f),
      Vec3f(0, -1, 0) // 朝下
  });
  // 第二个光源
  lights.push_back({
      Vec3f(-0.5f, 1.5f, 0.0f), // 位置
      Vec3f(3.0f, 5.0f, 5.0f),  // 辐射通量
      Vec2f(0.5f, 0.5f),        // 尺寸
      Vec3f(0, -1, 0)           // 朝下
  });

  // 采样数
  int shadow_samples = 128;

  // 遍历每个光源
  for (const auto &light : lights) {

    Vec3f current_light_contribution(0.0f);

    // 计算当前光源的几何向量
    Vec3f light_u = Vec3f(1, 0, 0) * light.size.x / 2.0f;
    Vec3f light_v = Vec3f(0, 0, 1) * light.size.y / 2.0f;
    Float area = light.size.x * light.size.y;

    // 计算辐射亮度 Le = Flux / (Area * PI)
    Vec3f Le = light.flux / (area * PI);

    for (int i = 0; i < shadow_samples; ++i) {
      Vec2f uv = sampler.get2D();

      // 在矩形内随机取点
      Vec3f light_sample_pos = light.center + (2.0f * uv.x - 1.0f) * light_u +
                               (2.0f * uv.y - 1.0f) * light_v;

      Vec3f diff = light_sample_pos - interaction.p;

      Float dist_sq_real = Dot(diff, diff);
      if (dist_sq_real < 1e-6f)
        continue;

      // 归一化
      Float dist = std::sqrt(dist_sq_real);
      Vec3f light_dir = diff / dist;

      Float dist_sq_clamped = std::max(dist_sq_real, 0.01f);

      Float cos_theta_surf = Dot(interaction.normal, light_dir);
      if (cos_theta_surf <= 0)
        continue;

      // 阴影遮挡测试
      auto shadow_ray = DifferentialRay(interaction.p, light_dir);
      SurfaceInteraction shadow_interaction;
      bool occluded = scene->intersect(shadow_ray, shadow_interaction);

      if (occluded &&
          Norm(shadow_interaction.p - interaction.p) < dist - 1e-4f) {
        continue; // 被遮挡
      }

      interaction.wi = light_dir;
      Vec3f f_r = interaction.bsdf->evaluate(interaction);

      // 使用 clamped 距离计算衰减
      Vec3f sample_val = Le * f_r * cos_theta_surf * area / dist_sq_clamped;

      current_light_contribution += sample_val;
    }

    total_color += current_light_contribution / (Float)shadow_samples;
  }

  return total_color;
}

/* ===================================================================== *
 *
 * Path Integrator's Implementation
 *
 * ===================================================================== */

void PathIntegrator::render(ref<Camera> camera, ref<Scene> scene) {
  // This is left as the next assignment
  UNIMPLEMENTED;
}

Vec3f PathIntegrator::Li(ref<Scene> scene, DifferentialRay &ray,
                         Sampler &sampler) const {
  // This is left as the next assignment
  UNIMPLEMENTED;
}

Vec3f PathIntegrator::directLighting(ref<Scene> scene,
                                     SurfaceInteraction &interaction,
                                     Sampler &sampler) const {
  // This is left as the next assignment
  UNIMPLEMENTED;
}

/* ===================================================================== *
 *
 * New Integrator's Implementation
 *
 * ===================================================================== */

// Instantiate template
// clang-format off
template Vec3f
IncrementalPathIntegrator::Li<Path>(ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const;
template Vec3f
IncrementalPathIntegrator::Li<PathImmediate>(ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const;
// clang-format on

// This is exactly a way to separate dec and def
template <typename PathType>
Vec3f IncrementalPathIntegrator::Li( // NOLINT
    ref<Scene> scene, DifferentialRay &ray, Sampler &sampler) const {
  // This is left as the next assignment
  UNIMPLEMENTED;
}

RDR_NAMESPACE_END
