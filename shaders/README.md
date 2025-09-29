# Vendored Shaders

These are the vulkan shaders vendored in from the 1.1.4 release of the FidelityFX SDK. As there may be some slight modifications made to make these shaders work with wgpu/webgpu, we vendor them in fully. To re-vendor these shaders, you can perform the following operations. By doing this, you will be able to see all modifications made to the shaders.

```bash
cargo xtask vendor
cp -r ffx/sdk/src/backends/vk/shaders/* ./shaders/src/
cp -r ffx/sdk/include/FidelityFX/gpu/* ./shaders/include/
rm ./shaders/include/CMakeCompileShaders.txt
```
