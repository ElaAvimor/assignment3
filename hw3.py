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
    epsilon = 0.001  # Small bias to prevent self-shadowing
    direction_to_light = normalize(light.get_light_ray(p).direction)
    biased_hit_point = p + epsilon * direction_to_light
    hit_to_light_ray = Ray(biased_hit_point, direction_to_light)
    hit_to_light_dist = light.get_distance_from_light(p)
    dist, nearest_obj, _ = hit_to_light_ray.nearest_intersected_object(objects)
    if nearest_obj is None or dist >= hit_to_light_dist:
        return 1
    return 0
   
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

    # Pikachu
    # Body
    body = Sphere([0, 0, -2], 1)
    body.set_material([1, 1, 0], [1, 1, 0], [0.5, 0.5, 0.3], 150, 0.1)
    objects.append(body)

    # Ears
    # ear1 = Pyramid([
    #     [0.6, 0.8, -1.5], [0.9, 0.8, -1.5], [0.75, 1.5, -2]  # Adjust coordinates as needed
    # ])
    # ear1.set_material([0, 0, 0], [0, 0, 0], [0, 0, 0], 10, 0.1)
    # objects.append(ear1)

    # ear2 = Pyramid([
    #     [-0.6, 0.8, -1.5], [-0.9, 0.8, -1.5], [-0.75, 1.5, -2]  # Adjust coordinates as needed
    # ])
    # ear2.set_material([0, 0, 0], [0, 0, 0], [0, 0, 0], 10, 0.1)
    # objects.append(ear2)

    # Pokeball
    # Bottom half
    bottom_half = Sphere([2, 0, -1], 0.5)
    bottom_half.set_material([1, 1, 1], [1, 1, 1], [1, 1, 1], 100, 0.2)
    objects.append(bottom_half)

    # Top half
    top_half = Sphere([2, 0, -0.5], 0.5)
    top_half.set_material([1, 0, 0], [1, 0, 0], [1, 0, 0], 100, 0.2)
    objects.append(top_half)

    # Center button
    center_button = Sphere([2, 0, -0.75], 0.1)
    center_button.set_material([1, 1, 1], [1, 1, 1], [1, 1, 1], 100, 0.5)
    objects.append(center_button)

    # Ground plane
    plane = Plane([0, 1, 0], [0, -1, 0])
    plane.set_material([0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [1, 1, 1], 1000, 0.5)
    objects.append(plane)

    # Lights
    # Directional light simulating sunlight
    light1 = DirectionalLight([1, 1, 1], direction=[-1, -0.5, -0.5])
    lights.append(light1)

    # Ambient light for softer shadows
    light2 = PointLight([0.5, 0.5, 0.5], position=[2, 2, 0], kc=0.1, kl=0.1, kq=0.01)
    lights.append(light2)

    camera = np.array([0,0,1])

    return camera, lights, objects 