# Setup

This document showcases how you could setup your AABB tree to suit your needs.

## Lean and mean

This approach uses the defaults provided by the abby library, this is the simplest approach and should be sufficient for most users. The default precision used is
`double` and the tree will use the `abby::vector2` class as the vector type. By default
a thickness factor is used when inserting AABBs.

```C++
  abby::tree<int> tree;
```

## Custom precision

The second template parameter of `abby::tree` is the vector representation type. It should
probably be one of `float` or `double` (possibly even `long double`).

```C++
  abby::tree<int, float> tree;
```

## Custom vector type

Since everyone rolls their own custom vector type in their games, it can be very handy to not have to convert vector types everywhere in your game. As such, it's possible to specify the vector type that the `abby::tree` will use. Such a type must feature public members `x` and `y`, along with a public member alias `value_type`. It also needs to support `operator+` and `operator-`, these criteria shouldn't be an issue for a decent vector implementation.

```C++
  struct my_vector {
    using value_type = float;
    value_type x;
    value_type y;

    my_vector operator+(const my_vector& other) const;
    my_vector operator-(const my_vector& other) const;
  };

  abby::tree<int, float, my_vector> tree;
```

## Thickness factor

When inserting an AABB, the default behaviour of `abby::tree` is to sligthly "fatten" the supplied AABB, in order to avoid having to modify the tree too much when AABBs are updated. This "thickness" factor can be tweaked or even completely disabled.

```C++
  abby::tree<int> tree; // Uses thickness factor of 0.05 by default

  // Increases thickness factor slightly
  tree.set_thickness_factor(0.08);

  // Disables any fattening of inserted AABBs
  tree.set_thickness_factor(std::nullopt); 
```
