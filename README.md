# abby [![Build status](https://ci.appveyor.com/api/projects/status/p0ej0hg4cmemaeau?svg=true)](https://ci.appveyor.com/project/AlbinJohansson/abby) [![Build Status](https://travis-ci.org/albin-johansson/abby.svg?branch=dev)](https://travis-ci.org/albin-johansson/abby) ![version](https://img.shields.io/badge/version-0.3.0-blue.svg) [![Language](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.wikipedia.org/wiki/C%2B%2B#Standardization) [![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)

A header-only implementation of an AABB tree.

## Purpose

Collision detection is common in many applications, especially games. Subsequently, collision
detection often require checking for potential collision between many different game objects, and
as such it is important that the collision detection is efficient. A naive implementation will end
up with quadratic complexity, which isn't very scalable. By using an AABB tree, bounding hitboxes
are stored in a tree according to their position and size, which in turn enables logarithmic
complexity for collision detection.

## Example

```C++

  #include <abby.hpp>
  #include <iterator> // back_inserter
  #include <vector>   // vector
  #include <iostream> // clog

  void foo()
  {
    // Constructs an AABB tree that uses integers as identifiers and AABBs with double precision
    abby::tree<int, double> tree;

    // Inserts a few AABBs, takes a lower and upper bound
    tree.insert(1, {150.0, 165.0}, {50.0, 50.0});
    tree.insert(2, {88.0, 63.0}, {150.0, 343.0});
    tree.insert(3, {412.0, 132.0}, {566.0, 291.0});

    // Could also use a stack buffer
    std::vector<int> candidates;  

    // Find collision candidates
    tree.query(1, std::back_inserter(candidates));  
    for (const auto candidate : candidates) {
      // Obtains an AABB
      const auto& aabb = tree.get_aabb(candidate);
    }
  
    // Prints the tree using a stream
    tree.print(std::clog);

    // Updates an existing AABB
    tree.update(2, {33.0, 76.0}, {123.0, 155.0});

    // Sets the position of an AABB
    tree.relocate(2, {12.0, 34.0});

    // Removes an AABB from the tree
    tree.erase(2);

    // Rebuild as optimal tree
    tree.rebuild();

    // Clear the tree of all entries
    tree.clear();
  }
```

## Acknowledgements

This library is an adapted and improved version of the [AABBCC](https://github.com/lohedges/aabbcc)
library. Furthermore, the AABB tree in the [Simple Voxel Engine](https://github.com/JamesRandall/SimpleVoxelEngine)
project also influenced the design of this library.
