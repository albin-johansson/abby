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
 */

#include <cassert>   // assert
#include <map>       // map
#include <optional>  // optional
#include <vector>    // vector

#ifndef ABBY_HEADER_GUARD
#define ABBY_HEADER_GUARD

namespace abby {

using opt_int = std::optional<int>;

template <typename T>
struct vec2 final
{
  T x{};
  T y{};
};

template <typename T = float>
struct aabb final
{
  vec2<T> min;
  vec2<T> max;
};

/**
 * \struct aabb_node
 *
 * \brief Represents a node in an AABB tree.
 *
 * \details Contains an AABB and the entity associated with the AABB, along
 * with tree information.
 *
 * \headerfile abby.hpp
 */
template <typename T, typename U = float>
struct aabb_node final
{
  T id;
  aabb<U> box;
  opt_int parent;
  opt_int left;
  opt_int right;
  opt_int next;
};

template <typename Key, typename T = float>
class aabb_tree final
{
 public:
  using key_type = Key;
  using size_type = std::size_t;
  using vector_type = vec2<T>;
  using aabb_type = aabb<T>;
  using node_type = aabb_node<key_type, T>;
  using index_type = int;

  void insert(const key_type& key, const aabb_type& box)
  {}

  void remove(const key_type& key)
  {}

  void update(const key_type& key, const aabb_type& box)
  {}

  void move(const key_type& key, const vector_type& offset)
  {}

  template <typename OutputIterator>
  void query_collisions(const key_type& key, OutputIterator iterator) const
  {}

  [[nodiscard]] auto get_aabb(const key_type& key) const -> const aabb_type&
  {
    return m_nodes.at(m_indexMap.at(key)).box;
  }

  [[nodiscard]] auto size() const noexcept -> size_type
  {
    return m_indexMap.size();
  }

  [[nodiscard]] auto is_empty() const noexcept -> bool
  {
    return m_indexMap.empty();
  }

 private:
  using opt_index = std::optional<index_type>;

  std::map<key_type, index_type> m_indexMap;
  std::vector<aabb_type> m_nodes;

  opt_index m_rootIndex{};
  opt_index m_nextFreeNodeIndex{};

  size_type m_allocatedNodes{};
  size_type m_nodeCapacity{24};
  size_type m_growthSize{m_nodeCapacity};

  void grow_pool()
  {
    assert(m_allocatedNodes == m_nodeCapacity);

    m_nodeCapacity += m_growthSize;
    m_nodes.resize(m_nodeCapacity);

    for (index_type index = m_allocatedNodes; index < m_nodeCapacity; ++index) {
      auto& node = m_nodes.at(index);
      node.next = index + 1;
    }

    m_nodes.at(m_nodeCapacity - 1).next.reset();
    m_nextFreeNodeIndex = m_allocatedNodes;
  }

  [[nodiscard]] auto allocate_node() -> index_type
  {
    // if we have no free tree nodes then grow the pool
    if (!m_nextFreeNodeIndex.has_value()) {
      grow_pool();
    }

    const auto index = m_nextFreeNodeIndex.value();
    const auto& node = m_nodes.at(index);
    m_nextFreeNodeIndex = node.next;
    ++m_allocatedNodes;

    return index;
  }

  void deallocate_node(index_type nodeIndex)
  {
    auto& node = m_nodes.at(nodeIndex);
    node.next = m_nextFreeNodeIndex;
    m_nextFreeNodeIndex = nodeIndex;
    --m_allocatedNodes;
  }

  void fix_upwards_tree(opt_index nodeIndex)
  {
    while (nodeIndex) {
      auto& node = m_nodes.at(nodeIndex.value());

      // every node should be a parent
      assert(node.left);
      assert(node.right);

      // fix height and area
      const auto& left = m_nodes.at(node.left.value());
      const auto& right = m_nodes.at(node.right.value());
      node.box = merge(left.box, right.box);

      nodeIndex = node.parent;
    }
  }

  [[nodiscard]] auto find_best_insertion_position(
      const node_type& leafNode) const -> index_type
  {
    auto treeNodeIndex = m_rootIndex.value();
    while (!is_leaf(m_nodes.at(treeNodeIndex))) {
      // because of the test in the while loop above we know we are never a leaf
      // inside it
      const auto& treeNode = m_nodes.at(treeNodeIndex);
      const auto leftNodeIndex = treeNode.left.value();
      const auto rightNodeIndex = treeNode.right.value();
      const auto& leftNode = m_nodes.at(leftNodeIndex);
      const auto& rightNode = m_nodes.at(rightNodeIndex);

      const auto combined = merge(treeNode.box, leafNode.box);

      const auto newParentNodeCost = 2.0f * combined.area;
      const auto minimumPushDownCost =
          2.0f * (combined.area - treeNode.box.area);

      // use the costs to figure out whether to create a new parent here or
      // descend
      const auto costLeft =
          get_left_cost(leftNode, leafNode, minimumPushDownCost);
      const auto costRight =
          get_right_cost(rightNode, leafNode, minimumPushDownCost);

      // if the cost of creating a new parent node here is less than descending
      // in either direction then we know we need to create a new parent node
      // here and attach the leaf to that
      if (newParentNodeCost < costLeft && newParentNodeCost < costRight) {
        break;
      }

      // otherwise descend in the cheapest direction
      if (costLeft < costRight) {
        treeNodeIndex = leftNodeIndex;
      } else {
        treeNodeIndex = rightNodeIndex;
      }
    }

    return treeNodeIndex;
  }

  void insert_leaf(index_type leaf)
  {}

  void remove_leaf(index_type leaf)
  {}

  void update_leaf(index_type leaf, const aabb_type& box)
  {}
};

}  // namespace abby

#endif  // ABBY_HEADER_GUARD
