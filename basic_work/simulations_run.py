import mitsuba as mi

mi.set_variant('scalar_rgb')

img = mi.render(mi.load_dict(mi.cornell_box()))

mi.Bitmap(img).write('cbox.exr')

import os
from os.path import realpath, join

import drjit as dr
import mitsuba as mi

#mi.set_variant('llvm_ad_rgb')
mi.set_variant('cuda_ad_rgb')

SCENE_DIR = realpath('../scenes')

from glob import glob
img_names = glob('aperture/*') + glob('focal/*')

for img_name in img_names:
    CONFIGS = {
        'wave': {
            'emitter': 'gray',
            'reference': join(SCENE_DIR, 'references/wave-1024.jpg'),
        },
        'sunday': {
            'emitter': 'bayer',
            'reference': img_name,
        },
    }

    # Pick one of the available configs
    config_name = 'sunday'
    # config_name = 'wave'

    config = CONFIGS[config_name]
    print('[i] Reference image selected:', config['reference'])
    mi.Bitmap(config['reference'])

    if 'PYTEST_CURRENT_TEST' not in os.environ:
        config.update({
            'render_resolution': (128, 128),
            'heightmap_resolution': (512, 512),
            'n_upsampling_steps': 4,
            'spp': 32,
            'max_iterations': 2000,
            'learning_rate': 3e-5,
        })
    else:
        # IGNORE THIS: When running under pytest, adjust parameters to reduce computation time
        config.update({
            'render_resolution': (64, 64),
            'heightmap_resolution': (128, 128),
            'n_upsampling_steps': 0,
            'spp': 8,
            'max_iterations': 50,
            'learning_rate': 3e-5,
        })

    config_name = img_name.split('/')[-1].split('.')[0]

    output_dir = realpath(join('.', 'outputs', config_name))
    os.makedirs(output_dir, exist_ok=True)
    print('[i] Results will be saved to:', output_dir)

    mi.Thread.thread().file_resolver().append(SCENE_DIR)

    def create_flat_lens_mesh(resolution):
        # Generate UV coordinates
        U, V = dr.meshgrid(
            dr.linspace(mi.Float, 0, 1, resolution[0]),
            dr.linspace(mi.Float, 0, 1, resolution[1]),
            indexing='ij'
        )
        texcoords = mi.Vector2f(U, V)

        # Generate vertex coordinates
        X = 2.0 * (U - 0.5)
        Y = 2.0 * (V - 0.5)
        vertices = mi.Vector3f(X, Y, 0.0)

        # Create two triangles per grid cell
        faces_x, faces_y, faces_z = [], [], []
        for i in range(resolution[0] - 1):
            for j in range(resolution[1] - 1):
                v00 = i * resolution[1] + j
                v01 = v00 + 1
                v10 = (i + 1) * resolution[1] + j
                v11 = v10 + 1
                faces_x.extend([v00, v01])
                faces_y.extend([v10, v10])
                faces_z.extend([v01, v11])

        # Assemble face buffer
        faces = mi.Vector3u(faces_x, faces_y, faces_z)

        # Instantiate the mesh object
        mesh = mi.Mesh("lens-mesh", resolution[0] * resolution[1], len(faces_x), has_vertex_texcoords=True)

        # Set its buffers
        mesh_params = mi.traverse(mesh)
        mesh_params['vertex_positions'] = dr.ravel(vertices)
        mesh_params['vertex_texcoords'] = dr.ravel(texcoords)
        mesh_params['faces'] = dr.ravel(faces)
        mesh_params.update()

        return mesh

    lens_res = config.get('lens_res', config['heightmap_resolution'])
    lens_fname = join(output_dir, 'lens_{}_{}.ply'.format(*lens_res))

    if not os.path.isfile(lens_fname):
        m = create_flat_lens_mesh(lens_res)
        m.write_ply(lens_fname)
        print('[+] Wrote lens mesh ({}x{} tesselation) file to: {}'.format(*lens_res, lens_fname))

    emitter = None
    if config['emitter'] == 'gray':
        emitter = {
            'type':'directionalarea',
            'radiance': {
                'type': 'spectrum',
                'value': 0.8
            },
        }
    elif config['emitter'] == 'bayer':
        bayer = dr.zeros(mi.TensorXf, (32, 32, 3))
        bayer[ ::2,  ::2, 2] = 2.2
        bayer[ ::2, 1::2, 1] = 2.2
        bayer[1::2, 1::2, 0] = 2.2

        emitter = {
            'type':'directionalarea',
            'radiance': {
                'type': 'bitmap',
                'bitmap': mi.Bitmap(bayer),
                'raw': True,
                'filter_type': 'nearest'
            },
        }


    integrator = {
        'type': 'ptracer',
        'samples_per_pass': 256,
        'max_depth': 4,
        'hide_emitters': False,
    }


    # Looking at the receiving plane, not looking through the lens
    sensor_to_world = mi.ScalarTransform4f().look_at(
        target=[0, -20, 0],
        origin=[0, -4.65, 0],
        up=[0, 0, 1]
    )
    resx, resy = config['render_resolution']
    sensor = {
        'type': 'perspective',
        'near_clip': 1,
        'far_clip': 1000,
        'fov': 45,
        'to_world': sensor_to_world,

        'sampler': {
            'type': 'independent',
            'sample_count': 512  # Not really used
        },
        'film': {
            'type': 'hdrfilm',
            'width': resx,
            'height': resy,
            'pixel_format': 'rgb',
            'rfilter': {
                # Important: smooth reconstruction filter with a footprint larger than 1 pixel.
                'type': 'gaussian'
            }
        },
    }


    scene = {
        'type': 'scene',
        'sensor': sensor,
        'integrator': integrator,
        # Glass BSDF
        'simple-glass': {
            'type': 'dielectric',
            'id': 'simple-glass-bsdf',
            'ext_ior': 'air',
            'int_ior': 1.5,
            'specular_reflectance': { 'type': 'spectrum', 'value': 0 },
        },
        'white-bsdf': {
            'type': 'diffuse',
            'id': 'white-bsdf',
            'reflectance': { 'type': 'rgb', 'value': (1, 1, 1) },
        },
        'black-bsdf': {
            'type': 'diffuse',
            'id': 'black-bsdf',
            'reflectance': { 'type': 'spectrum', 'value': 0 },
        },
        # Receiving plane
        'receiving-plane': {
            'type': 'obj',
            'id': 'receiving-plane',
            'filename': 'meshes/rectangle.obj',
            'to_world': \
                mi.ScalarTransform4f().look_at(
                    target=[0, 1, 0],
                    origin=[0, -7, 0],
                    up=[0, 0, 1]
                ).scale((5, 5, 5)),
            'bsdf': {'type': 'ref', 'id': 'white-bsdf'},
        },
        # Glass slab, excluding the 'exit' face (added separately below)
        'slab': {
            'type': 'obj',
            'id': 'slab',
            'filename': 'meshes/slab.obj',
            'to_world': mi.ScalarTransform4f().rotate(axis=(1, 0, 0), angle=90),
            'bsdf': {'type': 'ref', 'id': 'simple-glass'},
        },
        # Glass rectangle, to be optimized
        'lens': {
            'type': 'ply',
            'id': 'lens',
            'filename': lens_fname,
            'to_world': mi.ScalarTransform4f().rotate(axis=(1, 0, 0), angle=90),
            'bsdf': {'type': 'ref', 'id': 'simple-glass'},
        },

        # Directional area emitter placed behind the glass slab
        'focused-emitter-shape': {
            'type': 'obj',
            'filename': 'meshes/rectangle.obj',
            'to_world': mi.ScalarTransform4f().look_at(
                target=[0, 0, 0],
                origin=[0, 5, 0],
                up=[0, 0, 1]
            ),
            'bsdf': {'type': 'ref', 'id': 'black-bsdf'},
            'focused-emitter': emitter,
        },
    }


    scene = mi.load_dict(scene)

    def load_ref_image(config, resolution, output_dir):
        b = mi.Bitmap(config['reference'])
        b = b.convert(mi.Bitmap.PixelFormat.RGB, mi.Bitmap.Float32, False)
        if dr.any(b.size() != resolution):
            b = b.resample(resolution)

        mi.util.write_bitmap(join(output_dir, 'out_ref.exr'), b)

        print('[i] Loaded reference image from:', config['reference'])
        return mi.TensorXf(b)

    # Make sure the reference image will have a resolution matching the sensor
    sensor = scene.sensors()[0]
    crop_size = sensor.film().crop_size()
    image_ref = load_ref_image(config, crop_size, output_dir=output_dir)


    initial_heightmap_resolution = [r // (2 ** config['n_upsampling_steps'])
                                    for r in config['heightmap_resolution']]
    upsampling_steps = dr.square(dr.linspace(mi.Float, 0, 1, config['n_upsampling_steps']+1, endpoint=False).numpy()[1:])
    upsampling_steps = (config['max_iterations'] * upsampling_steps).astype(int)
    print('The resolution of the heightfield will be doubled at iterations:', upsampling_steps)

    heightmap_texture = mi.load_dict({
        'type': 'bitmap',
        'id': 'heightmap_texture',
        'bitmap': mi.Bitmap(dr.zeros(mi.TensorXf, initial_heightmap_resolution)),
        'raw': True,
    })

    # Actually optimized: the heightmap texture
    params = mi.traverse(heightmap_texture)
    params.keep(['data'])
    opt = mi.ad.Adam(lr=config['learning_rate'], params=params)


    params_scene = mi.traverse(scene)

    # We will always apply displacements along the original normals and
    # starting from the original positions.
    positions_initial = dr.unravel(mi.Vector3f, params_scene['lens.vertex_positions'])
    normals_initial   = dr.unravel(mi.Vector3f, params_scene['lens.vertex_normals'])

    lens_si = dr.zeros(mi.SurfaceInteraction3f, dr.width(positions_initial))
    lens_si.uv = dr.unravel(type(lens_si.uv), params_scene['lens.vertex_texcoords'])

    def apply_displacement(amplitude = 1.):
        # Enforce reasonable range. For reference, the receiving plane
        # is 7 scene units away from the lens.
        vmax = 1 / 100.
        params['data'] = dr.clip(params['data'], -vmax, vmax)
        dr.enable_grad(params['data'])

        height_values = heightmap_texture.eval_1(lens_si)
        new_positions = (height_values * normals_initial * amplitude + positions_initial)
        params_scene['lens.vertex_positions'] = dr.ravel(new_positions)
        params_scene.update()

    def scale_independent_loss(image, ref):
        """Brightness-independent L2 loss function."""
        scaled_image = image / dr.mean(dr.detach(image))
        scaled_ref = ref / dr.mean(ref)
        return dr.mean(dr.square(scaled_image - scaled_ref))

    import time
    start_time = time.time()
    mi.set_log_level(mi.LogLevel.Warn)
    iterations = config['max_iterations']
    loss_values = []
    spp = config['spp']

    for it in range(iterations):
        t0 = time.time()

        # Apply displacement and update the scene BHV accordingly
        apply_displacement()

        # Perform a differentiable rendering of the scene
        image = mi.render(scene, params, seed=it, spp=2 * spp, spp_grad=spp)

        # Scale-independent L2 function
        loss = scale_independent_loss(image, image_ref)

        # Back-propagate errors to input parameters and take an optimizer step
        dr.backward(loss)

        # Take a gradient step
        opt.step()

        # Increase resolution of the heightmap
        if it in upsampling_steps:
            opt['data'] = dr.upsample(opt['data'], scale_factor=(2, 2, 1))

        # Carry over the update to our "latent variable" (the heightmap values)
        params.update(opt)

        # Log progress
        elapsed_ms = 1000. * (time.time() - t0)
        current_loss = loss.array[0]
        loss_values.append(current_loss)
        mi.Thread.thread().logger().log_progress(
            it / (iterations-1),
            f'Iteration {it:03d}: loss={current_loss:g} (took {elapsed_ms:.0f}ms)',
            'Caustic Optimization', '')

        # Increase rendering quality toward the end of the optimization
        if it in (int(0.7 * iterations), int(0.9 * iterations)):
            spp *= 2
            opt.set_learning_rate(0.5 * opt.lr['data'])


    end_time = time.time()
    print(((end_time - start_time) * 1000) / iterations, ' ms per iteration on average')
    mi.set_log_level(mi.LogLevel.Info)

    mi.set_log_level(mi.LogLevel.Error)
    fname = join(output_dir, 'heightmap_final.exr')
    mi.util.write_bitmap(fname, params['data'])
    print('[+] Saved final heightmap state to:', os.path.basename(fname))

    fname = join(output_dir, 'lens_displaced.ply')
    apply_displacement()
    lens_mesh = [m for m in scene.shapes() if m.id() == 'lens'][0]
    lens_mesh.write_ply(fname)
    print('[+] Saved displaced lens to:', os.path.basename(fname))

    import matplotlib.pyplot as plt

    def show_image(ax, img, title):
        ax.imshow(mi.util.convert_to_bitmap(img))
        ax.axis('off')
        ax.set_title(title)


    def show_heightmap(fig, ax, values, title):
        im = ax.imshow(values.squeeze(), vmax=1e-4)
        fig.colorbar(im, ax=ax)
        ax.axis('off')
        ax.set_title(title)

    fig, ax = plt.subplots(2, 2, figsize=(11, 10))
    ax = ax.ravel()
    ax[0].plot(loss_values)
    ax[0].set_xlabel('Iteration'); ax[0].set_ylabel('Loss value'); ax[0].set_title('Convergence plot')

    show_heightmap(fig, ax[1], params['data'].numpy(), 'Final heightmap')
    show_image(ax[2], image_ref, 'Reference')
    show_image(ax[3], image,     'Final state')
    # plt.show()


    from PIL import Image
    import numpy as np

    # Extract the heightmap data
    heightmap_data = params['data'].numpy()

    # Normalize the values to the range [0, 255] for saving as an image
    heightmap_normalized = (255 * (heightmap_data.squeeze() / np.max(heightmap_data))).astype(np.uint8)

    # Convert to an image and save
    heightmap_image = Image.fromarray(heightmap_normalized)
    # heightmap_image.save('heightmap.png')
    heightmap_image.save(join(output_dir, 'heightmap.png'))

    filename = join(output_dir, 'recon_img.png')

    # Normalize the image for saving
    image_data = np.array(image)  # Ensure the image is a NumPy array
    normalized_image = (255 * (image_data.squeeze() / np.max(image_data))).astype(np.uint8)

    # Save the normalized image using PIL
    image_to_save = Image.fromarray(normalized_image)
    image_to_save.save(join(output_dir, filename))

    # filename = join(output_dir, 'loss_img.png')