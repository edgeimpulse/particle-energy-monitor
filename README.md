# Exported Edge Impulse's Particle Library for Imagine-Demo-All-Permutations_inferencing

## To use in Particle Workbench

1. In Workbench, select **Particle: Import Project** and select the `project.properties` file in the directory that you just downloaded and extracted.

1. Use **Particle: Configure Project for Device** and select **deviceOS@5.3.2** and choose a target. (e.g. **P2** , this option is also used for the Photon 2).

1. Compile with  **Particle: Compile application (local)**

1. Flash with **Particle: Flash application (local)**


> At this time you cannot use the **Particle: Cloud Compile** or **Particle: Cloud Flash** options; local compilation is required.

## Examples

`src/main.cpp` already contains one of the examples found in `examples/` directory.  If
you wish to use a different example, copy the `main.cpp` file from the example
in the `examples/` directory and replace `src/main.cpp`.

You may need to install additional libraries using **Particle: Install Library**
for some examples. See the example source for details.

### static_buffer

The `static_buffer` example can be used to feed raw features directly for
inference on target.  Copy raw features from the **Live classifiaction** page of
your project into the `features` array. For more information see
https://docs.edgeimpulse.com/docs/deployment/running-your-impulse-particle.
