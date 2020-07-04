/*
 * MIT License
 *
 * Copyright (c) 2020 Albin Johansson
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 * This code was adapted from the AABB.cc library, which can be found here:
 * https://github.com/lohedges/aabbcc. The AABB.cc library is licensed under
 * the Zlib license.
 */

#ifndef ABBY_HEADER_GUARD
#define ABBY_HEADER_GUARD

#include <entt.hpp>

/**
 * @namespace abby
 * @brief The top-level namespace that contains the components of the library.
 */
namespace abby {

/**
 * @struct vec2
 *
 * @brief Represents a simple floating-point 2-dimensional vector.
 *
 * @since 0.1.0
 *
 * @var vec2::x
 * The x-coordinate of the vector.
 *
 * @var vec2::y
 * The y-coordinate of the vector.
 *
 * @headerfile abby.hpp
 */
struct vec2 final {
  float x{};
  float y{};
};

/**
 * @struct aabb
 *
 * @brief Represents an AABB (Axis Aligned Bounding Box).
 *
 * @details An AABB is really just a fancy rectangle. They are mainly used
 * for collision detection systems, where trees can be built using AABBs in
 * order to decrease the complexity of finding potential collision candidates.
 * However, AABBs are not used for detailed collision detection. They are only
 * used to find potential collisions, which are then checked using more exact
 * collision detection systems.
 *
 * @note The "axis aligned" part is important, it means that the axes of the
 * rectangles must be aligned (parallel in relation to each other).
 * Otherwise, the systems that rely on AABBs won't work.
 *
 * @since 0.1.0
 *
 * @var aabb::min
 * The minimum x- and y-coordinates. Which are the coordinates of the
 * north-west corner of the box.
 *
 * @var aabb::max
 * The maximum x- and y-coordinates. Which are the coordinates of the
 * south-east corner of the box.
 *
 * @var aabb::center
 * The coordinates of the center point of the box.
 *
 * @var aabb::area
 * The area of the box.
 *
 * @headerfile abby.hpp
 */
struct aabb {
  vec2 min;
  vec2 max;
  vec2 center;
  float area{};
};

}  // namespace abby

#endif  // ABBY_HEADER_GUARD
