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
  using aabb_type = aabb<T>;
  using vector_type = vec2<T>;
  using index_type = int;

  void insert(const key_type& key, const aabb_type& box)
  {
    
  }

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
  std::map<key_type, index_type> m_indexMap;
  std::vector<aabb_type> m_nodes;

  std::optional<index_type> m_rootIndex{};
  std::optional<index_type> m_nextFreeNodeIndex{};

  size_type m_allocatedNodes{};
  size_type m_nodeCapacity{24};
  size_type m_growthSize{m_nodeCapacity};
};

}  // namespace abby

#endif  // ABBY_HEADER_GUARD
