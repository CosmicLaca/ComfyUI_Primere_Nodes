# Primere nodes for ComfyUI
## Workflow comparison table

- Normal Sampler (SD, SDXL): euler/karras, 20 steps, CFG: 8
- LCM Sampler (SD, SDXL): lcm/sgm_uniform, 8 steps, CFG: 1.18
- Turbo Sampler: euler_anchestral, 4 steps, CFG: 1
- SD model: Photon V1
- SDXL model: batchCOREALISMXL v40
- Turbo model: sd_xl_turbo 1.0 (Stability)
- GPU: RTX 3060 / 12 GB - 24GB RAM, Intel i5 3GHz. RTX 4090 / 16 running about half time.
- SD resolution: 768
- SDXL resolution: 1024
- Turbo resolution: 512
- Running time: 2nd run
- Upscaler: Ultimate SD upscaler. 4x for SD (768 x 4), 3.2x for SDXL (1024 x 3.2), 4x for Turbo (512 x 4). Always used and measured when available.
- Custom VAE always used (and loaded) if exist in workflow
- Refiners/detailers: face, hand, eye, mouth. always used if available in workflow
- Networks support Lora, Lycoris, Embedding and Hypernetworks for SD and SDXL
- All example images raw, no postprocesed, if upscaler or refiner/detailer available in workflow, then used and added to running time
- Click to image preview to visit original output size

<table>
    <tr><th>Workflow name</th><th>SD/SDXL</th><th>Dynamic</th><th>LCM</th><th>Turbo</th><th>Upscale<br>~6 mpx</th><th>Save</th><th>Refiners<br>4 refiners</th><th>Meta</th><th>Styles</th><th>Networks</th><th>VAE</th><th>RTX 3060/12</th><th>SD<br>Photon V1</th><th>SDXL<br>batchCOREALISMXL</th><th>SD LCM<br>Photon V1</th><th>SDXL LCM<br>batchCOREALISMXL</th><th>Turbo<br>sd_xl_turbo 1.0</th></tr>
    <tr>
        <td>Minimal</td>
        <td><img src="../readme_images/green-yes.png" height="30 px"></td>
        <td><img src="../readme_images/green-yes.png" height="30 px"></td>
        <td><img src="../readme_images/red-no.png" height="26px"></td>
        <td><img src="../readme_images/red-no.png" height="26px"></td>
        <td><img src="../readme_images/red-no.png" height="26px"></td>
        <td><img src="../readme_images/red-no.png" height="26px"></td>
        <td><img src="../readme_images/red-no.png" height="26px"></td>
        <td><img src="../readme_images/red-no.png" height="26px"></td>
        <td><img src="../readme_images/red-no.png" height="26px"></td>
        <td><img src="../readme_images/red-no.png" height="26px"></td>
        <td><img src="../readme_images/red-no.png" height="26px"></td>
        <td>SD: 9 sec<br>SDXL: 20 sec</td>
        <td><a href="../readme_images/example-minimal-sd-raw.jpg"><img src="../readme_images/example-minimal-sd.jpg" height="60 px"></a></td>
        <td><a href="../readme_images/example-minimal-sdxl-raw.jpg"><img src="../readme_images/example-minimal-sdxl.jpg" height="60 px"></a></td>
        <td><img src="../readme_images/red-no.png" height="26px"></td>
        <td><img src="../readme_images/red-no.png" height="26px"></td>
        <td><img src="../readme_images/red-no.png" height="26px"></td>
    </tr>
    <tr>
        <td>Basic</td>
        <td><img src="../readme_images/green-yes.png" height="30 px"></td>
        <td><img src="../readme_images/green-yes.png" height="30 px"></td>
        <td><img src="../readme_images/green-yes.png" height="30 px"></td>
        <td><img src="../readme_images/green-yes.png" height="30 px"></td>
        <td><img src="../readme_images/red-no.png" height="26px"></td>
        <td><img src="../readme_images/red-no.png" height="26px"></td>
        <td><img src="../readme_images/red-no.png" height="26px"></td>
        <td><img src="../readme_images/red-no.png" height="26px"></td>
        <td><img src="../readme_images/red-no.png" height="26px"></td>
        <td><img src="../readme_images/red-no.png" height="26px"></td>
        <td><img src="../readme_images/red-no.png" height="26px"></td>
        <td>SD: 9 sec<br>SD LCM: 7 sec<br>SDXL: 20 sec<br>SDXL LCM: 9 sec<br>Turbo: 2 sec</td>
        <td><a href="../readme_images/example-basic-sd-raw.jpg"><img src="../readme_images/example-basic-sd.jpg" height="60 px"></a></td>
        <td><a href="../readme_images/example-basic-sdxl-raw.jpg"><img src="../readme_images/example-basic-sdxl.jpg" height="60 px"></a></td>
        <td><a href="../readme_images/example-basic-sdlcm-raw.jpg"><img src="../readme_images/example-basic-sdlcm.jpg" height="60 px"></a></td>
        <td><a href="../readme_images/example-basic-sdxllcm-raw.jpg"><img src="../readme_images/example-basic-sdxllcm.jpg" height="60 px"></a></td>
        <td><a href="../readme_images/example-basic-turbo-raw.jpg"><img src="../readme_images/example-basic-turbo.jpg" height="60 px"></a></td>
    </tr>
    <tr>
        <td>Advanced</td>
        <td><img src="../readme_images/green-yes.png" height="30 px"></td>
        <td><img src="../readme_images/green-yes.png" height="30 px"></td>
        <td><img src="../readme_images/green-yes.png" height="30 px"></td>
        <td><img src="../readme_images/green-yes.png" height="30 px"></td>
        <td><img src="../readme_images/green-yes.png" height="30 px"></td>
        <td><img src="../readme_images/green-yes.png" height="30 px"></td>
        <td><img src="../readme_images/red-no.png" height="26px"></td>
        <td><img src="../readme_images/red-no.png" height="26px"></td>
        <td><img src="../readme_images/red-no.png" height="26px"></td>
        <td><img src="../readme_images/red-no.png" height="26px"></td>
        <td><img src="../readme_images/green-yes.png" height="30 px"></td>
        <td><u>Upscaled:</u><br>SD: 29 sec<br>SD LCM: 25 sec<br>SDXL: 67 sec<br>SDXL LCM: 52 sec<br>Turbo: 46 sec</td>
        <td><a href="../readme_images/example-advanced-sd-raw.jpg"><img src="../readme_images/example-advanced-sd.jpg" height="120 px"></a></td>
        <td><a href="../readme_images/example-advanced-sdxl-raw.jpg"><img src="../readme_images/example-advanced-sdxl.jpg" height="120 px"></a></td>
        <td><a href="../readme_images/example-advanced-sdlcm-raw.jpg"><img src="../readme_images/example-advanced-sdlcm.jpg" height="120 px"></a></td>
        <td><a href="../readme_images/example-advanced-sdxllcm-raw.jpg"><img src="../readme_images/example-advanced-sdxllcm.jpg" height="120 px"></a></td>
        <td><a href="../readme_images/example-advanced-turbo-raw.jpg"><img src="../readme_images/example-advanced-turbo.jpg" height="120 px"></a></td>
    </tr>
    <tr>
        <td>Latest</td>
        <td><img src="../readme_images/green-yes.png" height="30 px"></td>
        <td><img src="../readme_images/green-yes.png" height="30 px"></td>
        <td><img src="../readme_images/green-yes.png" height="30 px"></td>
        <td><img src="../readme_images/green-yes.png" height="30 px"></td>
        <td><img src="../readme_images/green-yes.png" height="30 px"></td>
        <td><img src="../readme_images/green-yes.png" height="30 px"></td>
        <td><img src="../readme_images/green-yes.png" height="30 px"></td>
        <td><img src="../readme_images/green-yes.png" height="30 px"></td>
        <td><img src="../readme_images/green-yes.png" height="30 px"></td>
        <td><img src="../readme_images/green-yes.png" height="30 px"></td>
        <td><img src="../readme_images/green-yes.png" height="30 px"></td>
        <td><u>Upscaled: 6 mpx</u><br>SD: 70 sec<br>SD LCM: 43 sec<br>SDXL: 215 sec<br>SDXL LCM: 120 sec<br>Turbo: 103 sec</td>
        <td><a href="../readme_images/example-latest-sd-raw.jpg"><img src="../readme_images/example-latest-sd.jpg" height="120 px"></a></td>
        <td><a href="../readme_images/example-latest-sdxl-raw.jpg"><img src="../readme_images/example-latest-sdxl.jpg" height="120 px"></a></td>
        <td><a href="../readme_images/example-latest-sdlcm-raw.jpg"><img src="../readme_images/example-latest-sdlcm.jpg" height="120 px"></a></td>
        <td><a href="../readme_images/example-latest-sdxllcm-raw.jpg"><img src="../readme_images/example-latest-sdxllcm.jpg" height="120 px"></a></td>
        <td><a href="../readme_images/example-latest-turbo-raw.jpg"><img src="../readme_images/example-latest-turbo.jpg" height="120 px"></a></td>
    </tr>
</table>