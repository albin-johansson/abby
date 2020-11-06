# abby [![Build status](https://ci.appveyor.com/api/projects/status/p0ej0hg4cmemaeau?svg=true)](https://ci.appveyor.com/project/AlbinJohansson/abby) ![version](https://img.shields.io/badge/version-0.1.0-blue.svg) [![Language](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B#Standardization) [![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A header-only implementation of an AABB tree.

## Purpose

Collision detection is common in many applications, especially games. Subsequently, collision detection often require checking for potential collision between many different game objects, and as such it is important that the collision detection is efficient. A naive implementation will end up with quadratic, i.e. O(n^2), complexity, which isn't very scalable. By using an AABB tree, bounding hitboxes are stored in a tree according to their position and size, which in turn enables logarithmic complexity for collision detection.

## Example

```C++

  #include <abby.hpp>
  #include <iterator> // back_inserter
  #include <vector>   // vector

  void foo()
  {
    // Constructs an AABB tree that uses integers as identifiers
    abby::aabb_tree<int> tree;

    // Inserts a few AABBs
    tree.insert(1, abby::make_aabb(abby::point{10, 10}, abby::size{120, 80}));
    tree.insert(2, abby::make_aabb(abby::point{88, 63}, abby::size{50, 43}));
    tree.insert(3, abby::make_aabb(abby::point{412, 132}, abby::size{66, 91}));

    std::vector<int> candidates; // Could also use a stack buffer!
    tree.query_collisions(1, std::back_inserter(candidates)); // Find collision candidates
    for (const auto candidate : candidates) {
      // Obtains an AABB
      const auto& aabb = tree.get_aabb(candidate);
    }

    // Replaces an AABB
    tree.replace(2, abby::make_aabb(abby::point{33, 76}, abby::size{123, 155}));

    // Adjusts the position of an AABB
    tree.set_position(2, abby::point{12, 34});

    // Removes an AABB from the tree
    tree.erase(2);
  }
```

## Acknowledgements

The implementation of the AABB tree was based on the AABB implementation in the Simple Voxel Engine project, which can be found [here](https://github.com/JamesRandall/SimpleVoxelEngine). It also uses the MIT license.
