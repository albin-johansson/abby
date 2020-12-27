# Queries

This document showcases various possible ways to query collisions from the AABB tree.

## abby::tree::query

This function takes an output iterator as a parameter, to which it will write any collision candidates. The simplest way to use this function is with a `std::vector` and `std::back_inserter` (located in the `<iterator>` standard header).

```C++
  abby::tree<int> tree;

  std::vector<int> candidates;
  tree.query(42, std::back_inserter(candidates));
  
  for (const auto candidate : candidates) {
    // ...
  }
```

However, the use of `std::vector` means that the collision candidates are stored on the heap. This is not great since dynamic memory allocations are very expensive, and since collision detection is performed so often in games we would like to avoid these allocations (or use a pool). This can be achieved relatively easily using the C++17 `std::pmr` containers (PMR stands for polymorphic memory resource). You'll need to include the `<memory_resource>` header for this.

```C++
  using aabb_tree = abby::tree<int>;
  aabb_tree tree;

  // This is our stack buffer, here we make large enough for 128 IDs
  std::array<std::byte, sizeof(aabb_tree::key_type) * 128> buffer;

  // This is our "memory resource", if we were to exceed our stack buffers 
  // capacity the default fallback is to allocate backup buffers on the heap
  std::pmr::monotonic_buffer_resource resource{buffer.data(),
                                               sizeof buffer};

  // Give our PMR vector a pointer to our memory resource
  std::pmr::vector<int> candidates{&resource}

  // We then call the query function just like we did with a std::vector
  tree.query(42, std::back_inserter(candidates));
  for (const auto candidate : candidates) {
    // ...
  }
```

## abby::tree::query_direct

If you don't want or need to store the collision candidates, you can simply supply a function object to the `query_direct` function. It works just like the `query` function 
except that the function object will be invoked for each collision candidate directly when
it is found. If you want to abort the query, you can return `true` (and return `false` to continue the query). However, this is optional, as you don't have to return anything in the function object.

```C++
  abby::tree<int> tree;

  // This query is exhaustive, the lambda is invoked for each candidate
  tree.query_direct(42, [](int candidate) {
    // ...  
  });

  // This query will stop when there are no more candidates or if the lambda returns true
  tree.query_direct(42, [](int candidate) {
    if (candidate == 7) {
      // ...
      return true;
    }    

    return false;
  });
```
