/*
 * MIT License
 *
 * Copyright (c) 2019-2020 Albin Johansson: adapted and improved source code
 * from the AABBCC and Simple Voxel Engine projects.
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
 * The code was based on two AABB tree implementations, one by James Randall
 * (Simple Voxel Engine) and another by Lester Hedges (AABBCC).
 *
 * The Simple Voxel Engine uses the MIT license:
 * https://github.com/JamesRandall/SimpleVoxelEngine.
 *
 * The AABBCC library uses the Zlib license: https://github.com/lohedges/aabbcc.
 */

#ifndef ABBY__HEADER__GUARD_
#define ABBY__HEADER__GUARD_

#include <algorithm>        // min, max
#include <array>            // array
#include <cassert>          // assert
#include <cstddef>          // byte
#include <deque>            // deque
#include <memory_resource>  // monotonic_buffer_resource
#include <optional>         // optional
#include <stack>            // stack
#include <unordered_map>    // unordered_map
#include <vector>           // vector

namespace abby {

/**
 * \struct vec2
 *
 * \brief A very simple representation of a 2D vector.
 *
 * \tparam T the representation type used.
 *
 * \since 0.1.0
 *
 * \see fvec2
 * \see dvec2
 *
 * \headerfile abby.hpp
 */
template <typename T>
struct vec2 final
{
  T x{};  ///< The x-axis component.
  T y{};  ///< The y-axis component.
};

/**
 * \typedef point
 *
 * \brief An alias for `vec2` that represents a point.
 *
 * \since 0.1.0
 */
template <typename T>
using point [[deprecated]] = vec2<T>;

/**
 * \typedef size
 *
 * \brief An alias for `vec2` that represents a width and height.
 *
 * \since 0.1.0
 */
template <typename T>
using size [[deprecated]] = vec2<T>;

// clang-format off
template <typename T> vec2(T, T) -> vec2<T>;
// clang-format on

/**
 * \brief Adds two vectors and returns the result.
 *
 * \tparam T the representation type used by the vectors.
 *
 * \param lhs the left-hand side vector.
 * \param rhs the right-hand side vector.
 *
 * \return a vector that is the result of adding the components of the two
 * vectors.
 *
 * \since 0.1.0
 */
template <typename T>
[[nodiscard]] constexpr auto operator+(const vec2<T>& lhs,
                                       const vec2<T>& rhs) noexcept -> vec2<T>
{
  return {lhs.x + rhs.x, lhs.y + rhs.y};
}

/**
 * \brief Subtracts two vectors and returns the result.
 *
 * \tparam T the representation type used by the vectors.
 *
 * \param lhs the left-hand side vector.
 * \param rhs the right-hand side vector.
 *
 * \return a vector that is the result of subtracting the components of the two
 * vectors.
 *
 * \since 0.1.0
 */
template <typename T>
[[nodiscard]] constexpr auto operator-(const vec2<T>& lhs,
                                       const vec2<T>& rhs) noexcept -> vec2<T>
{
  return {lhs.x - rhs.x, lhs.y - rhs.y};
}

/**
 * \brief Indicates whether or not two vectors are equal.
 *
 * \tparam T the representation type used by the vectors.
 *
 * \param lhs the left-hand side vector.
 * \param rhs the right-hand side vector.
 *
 * \return `true` if the two vectors are equal; `false` otherwise.
 *
 * \since 0.1.0
 */
template <typename T>
[[nodiscard]] constexpr auto operator==(const vec2<T>& lhs,
                                        const vec2<T>& rhs) noexcept -> bool
{
  return lhs.x == rhs.x && lhs.y == rhs.y;
}

/**
 * \brief Indicates whether or not two vectors aren't equal.
 *
 * \tparam T the representation type used by the vectors.
 *
 * \param lhs the left-hand side vector.
 * \param rhs the right-hand side vector.
 *
 * \return `true` if the two vectors aren't equal; `false` otherwise.
 *
 * \since 0.1.0
 */
template <typename T>
[[nodiscard]] constexpr auto operator!=(const vec2<T>& lhs,
                                        const vec2<T>& rhs) noexcept -> bool
{
  return !(lhs == rhs);
}

/**
 * \struct aabb
 *
 * \brief Represents an AABB (Axis Aligned Bounding Box).
 *
 * \note This is really just a glorified rectangle.
 *
 * \tparam T the representation type of the components used by the AABB.
 *
 * \since 0.1.0
 *
 * \see faabb
 * \see daabb
 *
 * \headerfile abby.hpp
 */
template <typename T = float>
struct aabb final
{
  vec2<T> min;  ///< The top-left corner of the AABB.
  vec2<T> max;  ///< The bottom-right corner of the AABB.

  /**
   * \brief Indicates whether or not the supplied AABB is contained within the
   * invoked AABB.
   *
   * \note The supplied AABB is still considered to be contained within the
   * invoked AABB if the borders of the inner AABB are overlapping the borders
   * of the outer AABB.
   *
   * \param other the AABB to check.
   *
   * \return `true` if the supplied AABB is contained in the AABB; `false`
   * otherwise.
   *
   * \since 0.1.0
   */
  [[nodiscard]] constexpr auto contains(const aabb<T>& other) const noexcept
      -> bool
  {
    if ((other.min.x < min.x) || (other.max.x > max.x) ||
        (other.min.y < min.y) || (other.max.y > max.y)) {
      return false;
    } else {
      return true;
    }
  }

  /**
   * \brief Indicates whether or not two AABBs are overlapping each other.
   *
   * \param other the other AABB to compare with.
   *
   * \return `true` if the two AABBs are overlapping each other; `false`
   * otherwise.
   *
   * \since 0.1.0
   */
  [[nodiscard]] constexpr auto overlaps(const aabb<T>& other) const noexcept
      -> bool
  {
    if ((other.max.x < min.x || other.min.x > max.x) ||
        (other.max.y < min.y || other.min.y > max.y)) {
      return false;
    } else {
      return true;
    }
  }

  /**
   * \brief Returns the area of the AABB.
   *
   * \note The area is not stored in the object, so it is computed for every
   * invocation of this function. This is of course not an expensive operation,
   * but it is worth knowing.
   *
   * \return the area of the AABB.
   *
   * \since 0.1.0
   */
  [[nodiscard]] constexpr auto area() const noexcept -> T
  {
    const auto width = max.x - min.x;
    const auto height = max.y - min.y;
    return width * height;
  }

  [[nodiscard]] constexpr auto center() const noexcept -> vec2<T>
  {
    const auto sum = min + max;
    return {sum.x / 2.0, sum / 2.0};
  }
};

// clang-format off
template <typename T> aabb(vec2<T>, vec2<T>) -> aabb<T>;
// clang-format on

/**
 * \brief Indicates whether or not two AABBs are equal.
 *
 * \tparam T the representation type used by the AABBs.
 *
 * \param lhs the left-hand side AABB.
 * \param rhs the right-hand side AABB.
 *
 * \return `true` if the two AABBs are equal; `false` otherwise.
 *
 * \since 0.1.0
 */
template <typename T>
[[nodiscard]] constexpr auto operator==(const aabb<T>& lhs,
                                        const aabb<T>& rhs) noexcept -> bool
{
  return lhs.min == rhs.min && lhs.max == rhs.max;
}

/**
 * \brief Indicates whether or not two AABBs aren't equal.
 *
 * \tparam T the representation type used by the AABBs.
 *
 * \param lhs the left-hand side AABB.
 * \param rhs the right-hand side AABB.
 *
 * \return `true` if the two AABBs aren't equal; `false` otherwise.
 *
 * \since 0.1.0
 */
template <typename T>
[[nodiscard]] constexpr auto operator!=(const aabb<T>& lhs,
                                        const aabb<T>& rhs) noexcept -> bool
{
  return !(lhs == rhs);
}

/**
 * \brief Creates and returns an AABB with the specified position and size.
 *
 * \details This is a convenience function for creating an AABB by specifying
 * the position and size, instead of the top-left and bottom-right corners.
 *
 * \tparam T the representation type used.
 *
 * \param position the position of the AABB (X and Y).
 * \param size the size of the AABB (width and height).
 *
 * \return an AABB at the specified position with the the specified size.
 *
 * \since 0.1.0
 */
template <typename T>
[[nodiscard]] constexpr auto make_aabb(const vec2<T>& position,
                                       const vec2<T>& size) -> aabb<T>
{
  aabb<T> box;

  box.min = position;
  box.max = box.min + size;

  return box;
}

/**
 * \brief Returns an AABB that is the union of the supplied pair of AABBs.
 *
 * \tparam T the representation type used by the AABBs.
 *
 * \param fst the first AABB.
 * \param snd the second AABB.
 *
 * \return an AABB that is the union of the two supplied AABBs.
 *
 * \since 0.1.0
 */
template <typename T>
[[nodiscard]] constexpr auto combine(const aabb<T>& fst,
                                     const aabb<T>& snd) noexcept -> aabb<T>
{
  aabb<T> result;

  result.min.x = std::min(fst.min.x, snd.min.x);
  result.min.y = std::min(fst.min.y, snd.min.y);

  result.max.x = std::max(fst.max.x, snd.max.x);
  result.max.y = std::max(fst.max.y, snd.max.y);

  return result;
}

/**
 * \struct aabb_node
 *
 * \brief Represents a node in an AABB tree.
 *
 * \details Contains an AABB and the entity associated with the AABB, along
 * with tree information.
 *
 * \since 0.1.0
 *
 * \headerfile abby.hpp
 */
template <typename T, typename U = float>
struct aabb_node final
{
  std::optional<T> id;
  aabb<U> box;
  std::optional<int> parent;
  std::optional<int> left;
  std::optional<int> right;
  std::optional<int> next;

  // Height of the node: 0 for leaves and `std::nullopt` for free nodes.
  std::optional<int> height;

  /**
   * \brief Indicates whether or not the node is a leaf node.
   *
   * \details A node is a leaf node if it has no left child node.
   *
   * \return `true` if the node is a leaf node; `false` otherwise.
   *
   * \since 0.1.0
   */
  [[nodiscard]] constexpr auto is_leaf() const noexcept -> bool
  {
    return !left;
  }
};

/// \cond FALSE

namespace detail {

template <typename T, typename U>
[[nodiscard]] constexpr auto get_left_cost(const aabb_node<T, U>& left,
                                           const aabb_node<T, U>& leaf,
                                           const double minimumCost) -> double
{
  if (left.is_leaf()) {
    return combine(leaf.box, left.box).area() + minimumCost;
  } else {
    const auto newLeftAabb = combine(leaf.box, left.box);
    return (newLeftAabb.area() - left.box.area()) + minimumCost;
  }
}

template <typename T, typename U>
[[nodiscard]] constexpr auto get_right_cost(const aabb_node<T, U>& right,
                                            const aabb_node<T, U>& leaf,
                                            const double minimumCost) -> double
{
  if (right.is_leaf()) {
    return combine(leaf.box, right.box).area() + minimumCost;
  } else {
    const auto newRightAabb = combine(leaf.box, right.box);
    return (newRightAabb.area() - right.box.area()) + minimumCost;
  }
}

}  // namespace detail

/// \endcond

template <typename T>
constexpr void fatten(aabb<T>& aabb, std::optional<double> factor) noexcept
{
  if (!factor) {
    return;
  }

  const auto size = aabb.max - aabb.min;

  aabb.min.x -= (*factor * size.x);
  aabb.min.y -= (*factor * size.y);
  aabb.max.x += (*factor * size.x);
  aabb.max.y += (*factor * size.y);
}

/**
 * \class tree
 *
 * \brief Represents a tree of AABBs used for efficient collision detection.
 *
 * \tparam Key the type of the keys associated with each node. Must be
 * comparable and preferable small and cheap to copy, e.g. `int`.
 * \tparam T the representation type used by the AABBs, should be a
 * floating-point type for best precision.
 *
 * \since 0.1.0
 *
 * \headerfile abby.hpp
 */
template <typename Key, typename T = float>
class tree final  // TODO revamp: relocate, query,
{
  template <typename U>
  using pmr_stack = std::stack<U, std::pmr::deque<U>>;

 public:
  using key_type = Key;
  using size_type = std::size_t;
  using vector_type = vec2<T>;
  using aabb_type = aabb<T>;
  using node_type = aabb_node<key_type, T>;
  using index_type = int;

  tree()
  {
    assert(!m_rootIndex);
    assert(m_nodeCount == 0);
    assert(m_nodeCapacity > 0);

    m_nodes.resize(m_nodeCapacity);
    for (auto i = 0; i < (m_nodeCapacity - 1); ++i) {
      auto& node = m_nodes.at(i);
      node.next = static_cast<index_type>(i + 1);
      node.height = std::nullopt;
    }

    auto& node = m_nodes.at(m_nodeCapacity - 1);
    node.next = std::nullopt;
    node.height = std::nullopt;

#ifndef NDEBUG
    validate();
#endif
  }

  /**
   * \brief Inserts an AABB in the tree.
   *
   * \pre `key` cannot be in use at the time of invoking this function.
   *
   * \param key the ID that will be associated with the box.
   * \param box the AABB that will be added.
   *
   * \since 0.1.0
   */
  void insert(const key_type& key, const aabb_type& box)
  {
    assert(!m_indexMap.count(key));  // Can't have same key multiple times!
    assert(box.area() > 0);

    const auto index = allocate_node();

    auto& node = m_nodes.at(index);
    node.box = box;
    node.id = key;
    node.height = 0;

    fatten(node.box, m_thicknessFactor);
    insert_leaf(index);

    m_indexMap.emplace(key, index);

#ifndef NDEBUG
    validate();
#endif
  }

  /**
   * \brief Adds an AABB to the tree.
   *
   * \note This function is equivalent to calling `insert` with the AABB
   * obtained from `make_aabb` using the supplied position and size.
   *
   * \pre `key` cannot be in use at the time of invoking this function.
   *
   * \param key the ID that will be associated with the AABB.
   * \param position the position of the AABB.
   * \param size the size of the AABB.
   *
   * \since 0.1.0
   */
  void emplace(const key_type& key,
               const vector_type& position,
               const vector_type& size)
  {
    insert(key, make_aabb(position, size));
  }

  /**
   * \brief Removes the AABB associated with the specified ID.
   *
   * \note This function has no effect if there is no AABB associated with the
   * specified ID.
   *
   * \param key the ID associated with the AABB that will be removed.
   *
   * \since 0.1.0
   */
  void erase(const key_type& key)
  {
    if (const auto it = m_indexMap.find(key); it != m_indexMap.end()) {
      const auto index = it->second;
      m_indexMap.erase(it);

      assert(index < m_nodeCapacity);
      assert(m_nodes.at(index).is_leaf());

      remove_leaf(index);
      free_node(index);
    }

#ifndef NDEBUG
    validate();
#endif
  }

  /**
   * \brief Replaces the AABB associated with the specified ID.
   *
   * \note This function has no effect if there is no AABB associated with the
   * specified ID.
   *
   * \param key the ID associated with the AABB that will be replaced.
   * \param box the new AABB that will be associated with the specified ID.
   * \param forceReinsert indicates whether or not the AABB is always
   * reinserted, which wont happen if this is set to `true` and the new AABB is
   * within the old AABB.
   *
   * \since 0.1.0
   */
  void replace(const key_type& key,
               const aabb_type& box,
               bool forceReinsert = false)
  {
    if (const auto it = m_indexMap.find(key); it != m_indexMap.end()) {
      const auto nodeIndex = it->second;

      assert(nodeIndex < m_nodeCapacity);
      assert(m_nodes.at(nodeIndex).is_leaf());

      // No need to update if the particle is still within its fattened AABB.
      if (!forceReinsert && m_nodes.at(nodeIndex).box.contains(box)) {
#ifndef NDEBUG
        validate();
#endif
        return;
      }

      auto copy = box;

      // Remove current leaf
      remove_leaf(nodeIndex);

      fatten(copy, m_thicknessFactor);
      m_nodes.at(nodeIndex).box = copy;

      insert_leaf(nodeIndex);
    }

#ifndef NDEBUG
    validate();
#endif
  }

  /**
   * \brief Updates the position of the AABB associated with the specified ID.
   *
   * \note This function has no effect if there is no AABB associated with the
   * specified ID.
   *
   * \param key the ID associated with the AABB that will be moved.
   * \param position the new position of the AABB associated with the specified
   * ID.
   *
   * \since 0.1.0
   */
  void relocate(const key_type& key, const vector_type& position)
  {
    if (const auto it = m_indexMap.find(key); it != m_indexMap.end()) {
      auto box = m_nodes.at(it->second).box;

      const auto size = box.max - box.min;
      box.min = position;
      box.max = box.min + size;

      replace(key, box);
    }

#ifndef NDEBUG
    validate();
#endif
  }

  /**
   * \brief Obtains collision candidates for the AABB associated with the
   * specified ID.
   *
   * \details In order to avoid unnecessary dynamic allocations, this function
   * returns the resulting collision candidates through an output iterator. This
   * means that it is possible to write collision candidates to both a stack
   * buffer and something like a `std::vector`.
   *
   * \details The output iterator can for instance be obtained using
   * `std::back_inserter`, if you're writing to a standard container.
   *
   * \tparam bufferSize the size of the initial stack buffer.
   * \tparam OutIterator the type of the output iterator.
   *
   * \param key the ID associated with the AABB to obtain collision candidates
   * for.
   * \param[out] iterator the output iterator used to write the collision
   * candidate IDs.
   *
   * \since 0.1.0
   */
  template <size_type bufferSize = 256, typename OutIterator>
  void query(const key_type& key, OutIterator iterator) const
  {
    if (const auto it = m_indexMap.find(key); it != m_indexMap.end()) {
      const auto& sourceNode = m_nodes.at(it->second);

      std::array<std::byte, sizeof(opt_index) * bufferSize> buffer;  // NOLINT
      std::pmr::monotonic_buffer_resource resource{buffer.data(),
                                                   sizeof buffer};

      pmr_stack<opt_index> stack{&resource};
      stack.push(m_rootIndex);
      while (!stack.empty()) {
        const auto nodeIndex = stack.top();
        stack.pop();

        if (!nodeIndex) {
          continue;
        }

        const auto& node = m_nodes.at(*nodeIndex);
        const auto copy = node.box;

        // Test for overlap between the AABBs
        if (node.box.overlaps(copy)) {
          // Check that we're at a leaf node
          if (node.is_leaf() && node.id) {
            // Can't interact with itself
            if (node.id != key) {
              *iterator = *node.id;
              ++iterator;
            }
          } else {
            stack.push(node.left);
            stack.push(node.right);
          }
        }
      }
    }
  }

  void set_fattening_factor(std::optional<double> factor) noexcept
  {
    m_thicknessFactor = factor;
  }

  /**
   * \brief Returns the AABB associated with the specified ID.
   *
   * \param key the ID associated with the desired AABB.
   *
   * \return the AABB associated with the specified ID.
   *
   * \throws if there is no AABB associated with the supplied ID.
   *
   * \since 0.1.0
   */
  [[nodiscard]] auto get_aabb(const key_type& key) const -> const aabb_type&
  {
    return m_nodes.at(static_cast<size_type>(m_indexMap.at(key))).box;
  }

  /**
   * \brief Returns the amount of AABBs stored in the tree.
   *
   * \note The returned value is not necessarily the amount of _nodes_ in the
   * tree.
   *
   * \return the current amount of AABBs stored in the tree.
   *
   * \since 0.1.0
   */
  [[nodiscard]] auto size() const noexcept -> size_type
  {
    return m_indexMap.size();
  }

  /**
   * \brief Indicates whether or not the tree is empty.
   *
   * \return `true` if there are no AABBs stored in the tree; `false` otherwise.
   *
   * \since 0.1.0
   */
  [[nodiscard]] auto is_empty() const noexcept -> bool
  {
    return m_indexMap.empty();
  }

 private:
  using opt_index = std::optional<index_type>;

  std::unordered_map<key_type, index_type> m_indexMap;
  std::vector<node_type> m_nodes;

  opt_index m_rootIndex;
  opt_index m_nextFreeNodeIndex{0};

  size_type m_nodeCount{0};
  size_type m_nodeCapacity{24};

  std::optional<double> m_thicknessFactor{0.05};

  /**
   * \brief Doubles the size of the node pool.
   *
   * \since 0.1.0
   */
  void grow_pool()
  {
    assert(m_nodeCount == m_nodeCapacity);

    m_nodeCapacity *= 2;  // We need more free nodes -> increase pool size
    m_nodes.resize(m_nodeCapacity);

    for (auto index = m_nodeCount; index < (m_nodeCapacity - 1); ++index) {
      auto& node = m_nodes.at(index);
      node.next = static_cast<index_type>(index + 1);
      node.height = std::nullopt;
    }

    auto& node = m_nodes.at(m_nodeCapacity - 1);
    node.next = std::nullopt;
    node.height = std::nullopt;

    // Update the index of the next free node
    m_nextFreeNodeIndex = static_cast<index_type>(m_nodeCount);
  }

  /**
   * \brief Returns the index to a new node.
   *
   * \details This function will grow the node pool if there are no available
   * nodes. Otherwise, this function will just increment the next free node
   * index and return the index of the previous next free node.
   *
   * \return the index of the allocated node.
   *
   * \since 0.1.0
   */
  [[nodiscard]] auto allocate_node() -> index_type
  {
    // if we have no free tree nodes then grow the pool
    if (!m_nextFreeNodeIndex) {
      grow_pool();
    }

    const auto index = m_nextFreeNodeIndex.value();  // Index of new node
    auto& node = m_nodes.at(index);

    m_nextFreeNodeIndex = node.next;

    node.parent = std::nullopt;
    node.left = std::nullopt;
    node.right = std::nullopt;
    node.height = 0;

    ++m_nodeCount;

    return index;
  }

  void free_node(const index_type nodeIndex)
  {
    assert(nodeIndex < m_nodeCapacity);
    assert(m_nodeCount > 0);

    auto& node = m_nodes.at(nodeIndex);
    node.next = m_nextFreeNodeIndex;
    node.height = std::nullopt;

    node.id = std::nullopt;
    node.right = std::nullopt;
    node.left = std::nullopt;
    node.parent = std::nullopt;

    m_nextFreeNodeIndex = nodeIndex;
    --m_nodeCount;
  }

  void rotate_right(const index_type nodeIndex,
                    const index_type leftIndex,
                    const index_type rightIndex)
  {
    auto& node = m_nodes.at(nodeIndex);
    auto& rightNode = m_nodes.at(rightIndex);

    const auto rightLeft = rightNode.left;
    const auto rightRight = rightNode.right;

    assert(rightLeft < m_nodeCapacity);
    assert(rightRight < m_nodeCapacity);

    // Swap node and its right-hand child
    rightNode.left = nodeIndex;
    rightNode.parent = node.parent;
    node.parent = rightIndex;

    // The node's old parent should now point to its right-hand child
    if (rightNode.parent) {
      auto& rightNodeParent = m_nodes.at(*rightNode.parent);
      if (rightNodeParent.left == nodeIndex) {
        rightNodeParent.left = rightIndex;
      } else {
        assert(rightNodeParent.right == nodeIndex);
        rightNodeParent.right = rightIndex;
      }
    } else {
      m_rootIndex = rightIndex;
    }

    // Rotate
    const auto& leftNode = m_nodes.at(leftIndex);
    auto& rightLeftNode = m_nodes.at(rightLeft.value());
    auto& rightRightNode = m_nodes.at(rightRight.value());

    if (rightLeftNode.height > rightRightNode.height) {
      rightNode.right = rightLeft;
      node.right = rightRight;

      rightRightNode.parent = nodeIndex;

      node.box = combine(leftNode.box, rightRightNode.box);
      rightNode.box = combine(node.box, rightLeftNode.box);

      node.height =
          1 + std::max(leftNode.height.value(), rightRightNode.height.value());
      rightNode.height =
          1 + std::max(node.height.value(), rightLeftNode.height.value());
    } else {
      rightNode.right = rightRight;
      node.right = rightLeft;

      rightLeftNode.parent = nodeIndex;
      node.box = combine(leftNode.box, rightLeftNode.box);
      rightNode.box = combine(node.box, rightRightNode.box);

      node.height =
          1 + std::max(leftNode.height.value(), rightLeftNode.height.value());
      rightNode.height =
          1 + std::max(node.height.value(), rightRightNode.height.value());
    }
  }

  void rotate_left(const index_type nodeIndex,
                   const index_type leftIndex,
                   const index_type rightIndex)
  {
    auto& node = m_nodes.at(nodeIndex);
    auto& leftNode = m_nodes.at(leftIndex);
    const auto leftLeft = leftNode.left;
    const auto leftRight = leftNode.right;

    assert(leftLeft < m_nodeCapacity);
    assert(leftRight < m_nodeCapacity);

    // Swap node and its left-hand child
    leftNode.left = nodeIndex;
    leftNode.parent = node.parent;
    node.parent = leftIndex;

    // The node's old parent should now point to its left-hand child
    if (leftNode.parent) {
      auto& leftNodeParent = m_nodes.at(*leftNode.parent);
      if (leftNodeParent.left == nodeIndex) {
        leftNodeParent.left = leftIndex;
      } else {
        assert(leftNodeParent.right == nodeIndex);
        leftNodeParent.right = leftIndex;
      }
    } else {
      m_rootIndex = leftIndex;
    }

    // Rotate
    const auto& rightNode = m_nodes.at(rightIndex);
    auto& leftLeftNode = m_nodes.at(leftLeft.value());
    auto& leftRightNode = m_nodes.at(leftRight.value());
    if (leftLeftNode.height > leftRightNode.height) {
      leftNode.right = leftLeft;
      node.left = leftRight;

      leftRightNode.parent = nodeIndex;

      node.box = combine(rightNode.box, leftRightNode.box);
      leftNode.box = combine(node.box, leftLeftNode.box);

      node.height =
          1 + std::max(rightNode.height.value(), leftRightNode.height.value());
      leftNode.height =
          1 + std::max(node.height.value(), leftLeftNode.height.value());
    } else {
      leftNode.right = leftRight;
      node.left = leftLeft;

      leftLeftNode.parent = nodeIndex;

      node.box = combine(rightNode.box, leftLeftNode.box);
      leftNode.box = combine(node.box, leftRightNode.box);

      node.height =
          1 + std::max(rightNode.height.value(), leftLeftNode.height.value());
      leftNode.height =
          1 + std::max(node.height.value(), leftRightNode.height.value());
    }
  }

  [[nodiscard]] auto balance(const index_type nodeIndex) -> index_type
  {
    const auto& node = m_nodes.at(nodeIndex);
    if (node.is_leaf() || node.height < 2) {
      return nodeIndex;
    }

    const auto leftIndex = node.left.value();
    const auto rightIndex = node.right.value();

    assert(leftIndex < m_nodeCapacity);
    assert(rightIndex < m_nodeCapacity);

    const auto& leftNode = m_nodes.at(leftIndex);
    const auto& rightNode = m_nodes.at(rightIndex);

    const auto currentBalance =
        rightNode.height.value() - leftNode.height.value();

    // Rotate right branch up
    if (currentBalance > 1) {
      rotate_right(nodeIndex, leftIndex, rightIndex);
      return rightIndex;
    }

    // Rotate left branch up
    if (currentBalance < -1) {
      rotate_left(nodeIndex, leftIndex, rightIndex);
      return leftIndex;
    }

    return nodeIndex;
  }

  void fix_tree_upwards(opt_index nodeIndex)
  {
    while (nodeIndex) {
      nodeIndex = balance(*nodeIndex);

      auto& node = m_nodes.at(*nodeIndex);
      const auto left = node.left;
      const auto right = node.right;

      assert(left);
      assert(right);

      const auto& leftNode = m_nodes.at(left.value());
      const auto& rightNode = m_nodes.at(right.value());

      node.height =
          1 + std::max(leftNode.height.value(), rightNode.height.value());
      node.box = combine(leftNode.box, rightNode.box);

      nodeIndex = node.parent;
    }
  }

  [[nodiscard]] auto find_best_sibling(const aabb_type& leafAabb) const
      -> index_type
  {
    auto index = m_rootIndex.value();

    while (!m_nodes.at(index).is_leaf()) {
      auto& node = m_nodes.at(index);
      const auto left = node.left.value();
      const auto right = node.right.value();

      assert(index != left);
      assert(index != right);

      const auto area = node.box.area();

      const auto combinedAabb = combine(node.box, leafAabb);
      const auto combinedArea = combinedAabb.area();

      // Cost of creating a new parent for this node and the new leaf.
      const auto cost = 2.0 * combinedArea;

      // Minimum cost of pushing the leaf further down the tree.
      const auto minimumCost = 2.0 * (combinedArea - area);

      const auto costLeft =
          detail::get_left_cost(m_nodes.at(left), node, minimumCost);
      const auto costRight =
          detail::get_right_cost(m_nodes.at(right), node, minimumCost);

      // Descend according to the minimum cost.
      if ((cost < costLeft) && (cost < costRight)) {
        break;
      }

      // Descend.
      if (costLeft < costRight) {
        index = left;
      } else {
        index = right;
      }
    }

    return index;
  }

  void insert_leaf(const index_type leafIndex)
  {
    if (m_rootIndex == std::nullopt) {
      // Tree was empty -> make the leaf the new root
      m_rootIndex = leafIndex;
      m_nodes.at(*m_rootIndex).parent = std::nullopt;
      return;
    }

    auto& leafNode = m_nodes.at(leafIndex);
    const auto leafAabb = leafNode.box;  // copy leaf nodes AABB
    const auto siblingIndex = find_best_sibling(leafAabb);
    auto& sibling = m_nodes.at(siblingIndex);

    // Create a new parent
    const auto oldParentIndex = sibling.parent;
    const auto newParentIndex = allocate_node();
    auto& newParent = m_nodes.at(newParentIndex);
    newParent.parent = oldParentIndex;
    newParent.box = combine(leafAabb, sibling.box);
    newParent.height = sibling.height.value() + 1;

    if (oldParentIndex != std::nullopt) {
      // The sibling was not the root
      auto& oldParent = m_nodes.at(oldParentIndex.value());
      if (oldParent.left == siblingIndex) {
        oldParent.left = newParentIndex;
      } else {
        oldParent.right = newParentIndex;
      }
    } else {
      // The sibling was the root
      m_rootIndex = newParentIndex;
    }

    newParent.left = siblingIndex;
    newParent.right = leafIndex;

    sibling.parent = newParentIndex;
    leafNode.parent = newParentIndex;

    // Walk up the tree and repair it
    fix_tree_upwards(leafNode.parent);
  }

  void adjust_ancestor_bounds(opt_index index)
  {
    while (index) {
      index = balance(*index);

      auto& node = m_nodes.at(*index);
      const auto left = node.left;
      const auto right = node.right;

      const auto& leftNode = m_nodes.at(left.value());
      const auto& rightNode = m_nodes.at(right.value());

      node.box = combine(leftNode.box, rightNode.box);
      node.height =
          1 + std::max(leftNode.height.value(), rightNode.height.value());

      index = node.parent;
    }
  }

  void remove_leaf(const index_type leafIndex)
  {
    if (leafIndex == m_rootIndex) {
      m_rootIndex = std::nullopt;
      return;
    }

    const auto parentIndex = m_nodes.at(leafIndex).parent.value();
    const auto& parentNode = m_nodes.at(parentIndex);
    const auto grandParentIndex = parentNode.parent;

    const auto sibling =
        (parentNode.left == leafIndex) ? parentNode.right : parentNode.left;

    // Destroy the parent and connect the sibling to the grandparent
    if (grandParentIndex) {
      auto& grandParent = m_nodes.at(*grandParentIndex);
      if (grandParent.left == parentIndex) {
        grandParent.left = sibling;
      } else {
        grandParent.right = sibling;
      }

      m_nodes.at(sibling.value()).parent = grandParentIndex;
      free_node(parentIndex);
      adjust_ancestor_bounds(grandParentIndex);
    } else {
      m_rootIndex = sibling;
      m_nodes.at(sibling.value()).parent = std::nullopt;
      free_node(parentIndex);
    }
  }

  void validate_structure(opt_index index) const
  {
    if (!index) {
      return;
    }

    if (index == m_rootIndex) {
      assert(m_nodes.at(*index).parent == std::nullopt);
    }

    const auto& node = m_nodes.at(*index);
    const auto left = node.left;
    const auto right = node.right;

    if (node.is_leaf()) {
      assert(left == std::nullopt);
      assert(right == std::nullopt);
      assert(node.height == 0);
    } else {
      assert(left < m_nodeCapacity);
      assert(right < m_nodeCapacity);
      assert(left != std::nullopt);
      assert(right != std::nullopt);

      const auto& leftNode = m_nodes.at(*left);
      const auto& rightNode = m_nodes.at(*right);

      assert(leftNode.parent == index);
      assert(rightNode.parent == index);

      validate_structure(left);
      validate_structure(right);
    }
  }

  void validate_metrics(opt_index index) const
  {
    if (!index) {
      return;
    }

    const auto& node = m_nodes.at(*index);
    const auto left = node.left;
    const auto right = node.right;

    if (node.is_leaf()) {
      assert(!left);
      assert(!right);
      assert(node.height == 0);
      return;
    } else {
      assert(left < m_nodeCapacity);
      assert(right < m_nodeCapacity);
      assert(left);
      assert(right);

      const auto& leftNode = m_nodes.at(*left);
      const auto& rightNode = m_nodes.at(*right);

      const auto height =
          1 + std::max(leftNode.height.value(), rightNode.height.value());
      assert(node.height == height);

      const auto aabb = combine(leftNode.box, rightNode.box);
      assert(aabb.min.x == node.box.min.x);
      assert(aabb.min.y == node.box.min.y);
      assert(aabb.max.x == node.box.max.x);
      assert(aabb.max.y == node.box.max.y);

      validate_metrics(left);
      validate_metrics(right);
    }
  }

  void validate()
  {
    validate_structure(m_rootIndex);
    validate_metrics(m_rootIndex);

    auto freeCount = 0;
    auto freeIndex = m_nextFreeNodeIndex;

    while (freeIndex) {
      assert(freeIndex < m_nodeCapacity);
      freeIndex = m_nodes.at(*freeIndex).next;
      ++freeCount;
    }

    //    assert(height() == computeHeight());
    assert((m_nodeCount + freeCount) == m_nodeCapacity);
  }
};

}  // namespace abby

#endif  // ABBY__HEADER__GUARD_
