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

        // 2. 根据采样位置生成光线
        auto ray =
            camera->generateDifferentialRay(pixel_sample.x, pixel_sample.y);

        // 3. 计算辐射度并累加到 Film
        // (直接取消原有代码的注释即可)
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
  Vec3f throughput(1.0); // 累积光通量 (用于处理有色玻璃，虽然这里全是白色的)

  // 记录是否找到了非透明的漫反射表面
  bool diffuse_found = false;
  SurfaceInteraction interaction;

  // 迭代追踪光线 (Trace the ray)
  for (int i = 0; i < max_depth; ++i) {
    // 1. 相交测试
    interaction = SurfaceInteraction();
    bool intersected = scene->intersect(ray, interaction);

    if (!intersected) {
      break; // 没打中任何东西，也就是打到了虚空(黑色背景)
    }

    // 2. 判断材质类型 (RTTI)
    bool is_ideal_diffuse =
        dynamic_cast<const IdealDiffusion *>(interaction.bsdf) != nullptr;
    bool is_perfect_refraction =
        dynamic_cast<const PerfectRefraction *>(interaction.bsdf) != nullptr;

    // 设置出射方向 (wo 指向光线来源)
    interaction.wo = -ray.direction;

    // --- 情况 A: 遇到折射物体 (Transparent/Refractive) ---
    if (is_perfect_refraction) {
      Vec3f wi;
      Float pdf;

      // 采样 BSDF 获得新的方向 wi (折射或全反射)
      // 注意: sample 函数会修改 interaction.wi
      Vec3f f = interaction.bsdf->sample(interaction, sampler, &pdf);

      // 更新 throughput (对于完美玻璃 f 通常是 1.0)
      throughput *= f;

      // 生成新光线: 原点在交点(处理了epsilon)，方向是新的 wi
      ray = interaction.spawnRay(interaction.wi);

      // 继续下一次循环 (Trace through)
      continue;
    }

    // --- 情况 B: 遇到漫反射物体 (Solid/Non-transparent) ---
    if (is_ideal_diffuse) {
      diffuse_found = true;
      break; // 找到了着色点，跳出循环去计算光照
    }

    // 其他情况 (忽略)
    break;
  }

  // 如果最终打到了漫反射物体，计算直接光照并乘以路径上的衰减(throughput)
  if (diffuse_found) {
    color = throughput * directLighting(scene, interaction);
  }

  return color;
}

Vec3f IntersectionTestIntegrator::directLighting(
    ref<Scene> scene, SurfaceInteraction &interaction) const {
  Vec3f total_color(0, 0, 0);
  struct SimpleLight {
    Vec3f position;
    Vec3f flux;
  };
  std::vector<SimpleLight> lights;
  lights.push_back({point_light_position, point_light_flux});
  // 在左侧添加一个红色的灯
  // lights.push_back({Vec3f(-0.5f, 1.5f, 0.0f), Vec3f(50.0f, 0.0f, 0.0f)});
  // 在右侧添加一个蓝色的灯
  // lights.push_back({Vec3f(2.0f, 3.0f, 0.0f), Vec3f(0.0f, 0.0f, 50.0f)});
  // TODO(HW3): Test for occlusion
  //
  // You should test if there is any intersection between interaction.p and
  // point_light_position using scene->intersect. If so, return an occluded
  // color. (or Vec3f color(0, 0, 0) to be specific)
  //
  // You may find the following variables useful:
  //
  // @see bool Scene::intersect(const Ray &ray, SurfaceInteraction &interaction)
  //    This function tests whether the ray intersects with any geometry in the
  //    scene. And if so, it returns true and fills the interaction with the
  //    intersection information.
  //
  //    You can use iteraction.p to get the intersection position.
  //
  for (const auto &light : lights) {

    // --- 以下逻辑全部针对当前 light 进行计算 ---

    // 注意：这里使用的是 light.position 而不是 point_light_position
    Float dist_to_light = Norm(light.position - interaction.p);
    Vec3f light_dir = Normalize(light.position - interaction.p);
    auto test_ray = DifferentialRay(interaction.p, light_dir);

    // --- 阴影测试 ---
    SurfaceInteraction shadow_interaction;
    bool occluded = scene->intersect(test_ray, shadow_interaction);

    if (occluded) {
      float dist_shadow = Norm(shadow_interaction.p - interaction.p);
      if (dist_shadow < dist_to_light - 1e-4f) {
        // [修改点 4] 关键修改：如果被遮挡，只是跳过当前这个光源，不要 return！
        continue;
      }
    }

    // --- 计算光照贡献 ---
    const BSDF *bsdf = interaction.bsdf;
    bool is_ideal_diffuse =
        dynamic_cast<const IdealDiffusion *>(bsdf) != nullptr;

    if (bsdf != nullptr && is_ideal_diffuse) {
      Float cos_theta = std::max(Dot(light_dir, interaction.normal), 0.0f);

      interaction.wi = light_dir;
      Vec3f f = bsdf->evaluate(interaction);

      // [修改点 5] 使用当前光源的 flux (light.flux)
      Vec3f intensity = light.flux * (1.0f / (4.0f * PI));

      // [修改点 6] 关键修改：使用 += 进行累加
      total_color +=
          f * intensity * cos_theta / (dist_to_light * dist_to_light);
    }
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
