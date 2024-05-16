from helper_classes import *
import matplotlib.pyplot as plt

def render_scene(camera, ambient, lights, objects, screen_size, max_depth):
    width, height = screen_size
    ratio = float(width) / height
    screen = (-1, 1 / ratio, 1, -1 / ratio)  # left, top, right, bottom

    image = np.zeros((height, width, 3))
    global_ambient = ambient

    for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
        for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
            # screen is on origin
            pixel = np.array([x, y, 0])
            origin = camera
            direction = normalize(pixel - origin)
            ray = Ray(origin, direction)

            color = np.zeros(3)

            # This is the main loop where each pixel color is computed.
            # done by us
            min_distance, nearest_object, p = ray.nearest_intersected_object(objects)
            if nearest_object:
                # print("Shape of global_ambient:", np.array(global_ambient), np.array(global_ambient).shape)
                # print("Shape of material_ambient:",np.array(nearest_object.ambient), np.array(nearest_object.ambient).shape)

                color += get_color(ray, nearest_object, global_ambient, lights, p, max_depth, objects)

            
            # We clip the values between 0 and 1 so all pixel values will make sense.
            image[i, j] = np.clip(color,0,1)

    return image


def get_color(ray, nearest_object, global_ambient, lights, p, level, objects):
    global_ambient = global_ambient
    material_ambient = nearest_object.ambient

    #print("Shape of global_ambient:", np.array(global_ambient).shape)
    #print("Shape of material_ambient:", np.array(nearest_object.ambient).shape)

    ambient_color = global_ambient * material_ambient
    diffuse_color = 0
    specular_color = 0
    reflective_color = 0

    for light in lights:
        shadow = return_if_light_shadow(light, p, objects)
        if shadow == 1:
            diffuse_color += calc_diffuse_color(light, nearest_object, p)
            specular_color += calc_specular_color(ray, light, nearest_object, p)

    level -= 1
    if level > 0:
        reflected_ray = construct_reflective_ray(ray, p, nearest_object)
        min_distance, reflected_obj, reflected_point = reflected_ray.nearest_intersected_object(objects)
        if reflected_obj:
            reflective_color = reflected_obj.reflection * get_color(reflected_ray, reflected_obj, global_ambient, lights,reflected_point, level, objects)

    color = ambient_color + diffuse_color + specular_color + reflective_color
    return color.astype(np.float64)

def calc_diffuse_color(light, nearest_object, p):
    if isinstance(nearest_object, Sphere):
        normal = normalize(p - nearest_object.center)
    else: 
        normal = nearest_object.normal

    light_vec = light.get_light_ray(p).direction
    dot_product = np.dot(normal, light_vec)
    dot_product = max(dot_product, 0)  # Ensure no negative lighting
    material_diffuse = np.array(nearest_object.diffuse) 
    light_intensity = light.get_intensity(p)

    return material_diffuse * light_intensity * dot_product


def calc_specular_color(ray, light, nearest_object, p):
    if isinstance(nearest_object, Sphere):
        normal = normalize(p - nearest_object.center)
    else:
         normal = nearest_object.normal

    light_vec = light.get_light_ray(p). direction
    reflect_dir = reflected(-light_vec, normal)
    view_dir = normalize(ray.origin - p)
    spec_angle = max(np.dot(reflect_dir, view_dir), 0)
    material_specular = np.array(nearest_object.specular)
    light_intensity = light.get_intensity(p)

    return material_specular * light_intensity * (spec_angle ** nearest_object.shininess)



def return_if_light_shadow(light, p, objects):
    light_ray = light.get_light_ray(p)
    min_distance, shadow_obj, light_intersection_point = light_ray.nearest_intersected_object(objects)

    if shadow_obj:
        distance_from_shadow = np.linalg.norm(p - light_intersection_point)
        if distance_from_shadow > light.get_distance_from_light(p):
            return 1
        else:
            return 0
    else:
        return 1
   
def construct_reflective_ray(ray, p, object):
    if isinstance(object, Sphere):
        normal = normalize(p - object.center)
    else:
        normal = normalize(object.normal)

    reflected_ray_vector = reflected(ray.direction, normal)
    return Ray(p, reflected_ray_vector)
    
# def construct_refractive_ray()

# Write your own objects and lights
# done by us
def your_own_scene():
    objects = []
    lights = []

    # Pikachu's Body
    pikachu_body = Sphere(center=[0, 0, -5], radius=1.7)
    pikachu_body.set_material(ambient=np.array([1, 1, 0]),  # Yellow ambient
                              diffuse=np.array([1, 1, 0]),  # Yellow diffuse
                              specular=np.array([1, 1, 0]),  # Yellow specular
                              shininess=50,
                              reflection=np.array([0.5, 0.5, 0.5]))
    objects.append(pikachu_body)
    # Pikachu's Eyes
    pikachu_eye_left = Sphere(center=[-0.6, 0.75, -3.5], radius=0.25)  # Moved closer to the front
    pikachu_eye_left.set_material(ambient=np.array([0, 0, 0]),
                                  diffuse=np.array([0, 0, 0]),
                                  specular=np.array([1, 1, 1]),
                                  shininess=100,
                                  reflection=np.array([0.5, 0.5, 0.5]))
    objects.append(pikachu_eye_left)

    pikachu_eye_right = Sphere(center=[0.6, 0.75, -3.5], radius=0.25)  # Moved closer to the front
    pikachu_eye_right.set_material(ambient=np.array([0, 0, 0]),
                                   diffuse=np.array([0, 0, 0]),
                                   specular=np.array([1, 1, 1]),
                                   shininess=100,
                                   reflection=np.array([0.5, 0.5, 0.5]))
    objects.append(pikachu_eye_right)

    # Pikachu's Ears
    # Left Ear
    vertices_ear_left = np.array([
        [-1.5, 3, -5],
        [-1.9, 2.2, -5],
        [-1.1, 2.2, -5],
        [-1.5, 2.5, -4.7],
        [-1.5, 2.5, -5.3]
    ])
    pikachu_ear_left = Pyramid(vertices_ear_left)
    pikachu_ear_left.set_material([0, 0, 0], [0, 0, 0], [0.3, 0.3, 0.3], 50, 0.5)
    pikachu_ear_left.apply_materials_to_triangles()
    objects.append(pikachu_ear_left)

    # Pikachu's Right Ear as a Pyramid
    vertices_ear_right = np.array([
        [1.5, 3, -5],
        [1.9, 2.2, -5],
        [1.1, 2.2, -5],
        [1.5, 2.5, -4.7],
        [1.5, 2.5, -5.3]
    ])
    pikachu_ear_right = Pyramid(vertices_ear_right)
    pikachu_ear_right.set_material([0, 0, 0], [0, 0, 0], [0.3, 0.3, 0.3], 50, 0.5)
    pikachu_ear_right.apply_materials_to_triangles()
    objects.append(pikachu_ear_right)

    # Pokeball
    pokeball_bottom = Sphere(center=[3, -1, -3], radius=0.5)
    pokeball_bottom.set_material(ambient=np.array([1, 1, 1]),
                                 diffuse=np.array([1, 1, 1]),
                                 specular=np.array([1, 1, 1]),
                                 shininess=300,
                                 reflection=np.array([0.5, 0.5, 0.5]))
    objects.append(pokeball_bottom)

    pokeball_top = Sphere(center=[3, -0.5, -3], radius=0.5)
    pokeball_top.set_material(ambient=np.array([1, 0, 0]),
                              diffuse=np.array([1, 0, 0]),
                              specular=np.array([1, 0, 0]),
                              shininess=300,
                              reflection=np.array([0.5, 0.5, 0.5]))
    objects.append(pokeball_top)
    # Ground plane
    plane = Plane(normal=[0, 1, 0], point=[0, -2.5, 0])
    plane.set_material(ambient=np.array([1, 1, 1]),  # White ambient
                       diffuse=np.array([1, 1, 1]),  # White diffuse
                       specular=np.array([0.5, 0.5, 0.5]),  # Mild specular to avoid too much shine
                       shininess=10,  # Lower shininess for a less glossy surface
                       reflection=np.array([0.1, 0.1, 0.1]))
    objects.append(plane)

    # Background plane - setting this to white
    background = Plane(normal=[0, 0, 1], point=[0, 0, -30])
    background.set_material(ambient=np.array([1, 1, 1]),  # White ambient
                            diffuse=np.array([1, 1, 1]),  # White diffuse
                            specular=np.array([0.5, 0.5, 0.5]),  # Mild specular
                            shininess=20,  # Moderate shininess to reflect some light
                            reflection=np.array([0.1, 0.1, 0.1]))
    objects.append(background)

    # Lighting setup
    directional_light = DirectionalLight(intensity=np.array([1.0, 1.0, 1.0]), direction=[1, -1, -1])
    point_light = PointLight(intensity=np.array([0.8, 0.8, 0.8]), position=[2, 2, -3], kc=1, kl=0.1, kq=0.01)
    lights.append(directional_light)
    lights.append(point_light)



    # Camera setup
    camera = np.array([0, 0, 1])

    return camera, lights, objects

