from helper_classes import *
import matplotlib.pyplot as plt

def render_scene(camera, ambient, lights, objects, screen_size, max_depth):
    width, height = screen_size
    ratio = float(width) / height
    screen = (-1, 1 / ratio, 1, -1 / ratio)  # left, top, right, bottom

    image = np.zeros((height, width, 3))

    for i, y in enumerate(np.linspace(screen[1], screen[3], height)):
        for j, x in enumerate(np.linspace(screen[0], screen[2], width)):
            # screen is on origin
            pixel = np.array([x, y, 0])
            origin = camera
            direction = normalize(pixel - origin)
            ray = Ray(origin, direction)

            color = np.zeros(3)
            min_distance, nearest_object, p = ray.nearest_intersected_object(objects)
            get_color(ray, nearest_object, ambient, lights, p)

            # This is the main loop where each pixel color is computed.


            
            # We clip the values between 0 and 1 so all pixel values will make sense.
            image[i, j] = np.clip(color,0,1)

    return image


def get_color(ray, nearest_object, ambient, lights, p, level):
    global_ambient = ambient
    material_ambient = nearest_object.ambient
    color = global_ambient * material_ambient
    for i in range(len(lights)):
        light = lights[i]
        light_shadowed = return_if_light_shadow(light, ray, p ,nearest_object)
        if light_shadowed == 1:
            color += calc_diffuse_color(ray ,light, nearest_object) + calc_specular_color(ray,light, nearest_object)

    level -= 1
    if level <= 0:
        return color
    else:
        reflected_ray = construct_reflective_ray(ray, p, nearest_object)
        reflected_obj,  = 


def calc_diffuse_color(light, nearest_object, p):
    if isinstance(nearest_object, Sphere):
        intersection_normal = normalize(p - object.center)
    else:
        intersection_normal = normalize(object.normal)

    material_diffuse = nearest_object.diffuse
    intersection_to_light = light.get_light_ray(p).direction
    light_intensity = light.get_intensity(p)

    return material_diffuse * light_intensity * (np.dot(intersection_normal,intersection_to_light))

def calc_specular_color(ray, light, nearest_object):
    if isinstance(nearest_object, Sphere):
        normal = normalize(p - object.center)
    else:
        normal = normalize(object.normal)

    intersection_to_light = light.get_light_ray(p).direction
    material_specular = nearest_object.specular
    intersection_to_eye = -1 * ray.direction
    intersection_to_reflected_light = normalize(reflected(-1*intersection_to_light, normal))
    light_intensity = light.get_intensity(nearest_object)

    return  material_specular * light_intensity * ((np.dot(intersection_to_eye,intersection_to_reflected_light))**nearest_object.shininess)


def return_if_light_shadow(light, ray, p ,nearest_object):
    light_ray = light.get_light_ray(p)
    shadow_obj = light_ray.nearest_intersected_object(nearest_object)[0]
    light_intersection_point = light_ray.nearest_intersected_object(nearest_object)[2]

    if not shadow_obj:
        return 1
    else:
        distance_from_shadow = np.linalg.norm(p - light_intersection_point)
        if distance_from_shadow > light.get_distance_from_light(p):
            return 1
        else:
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
# TODO
def your_own_scene():
    camera = np.array([0,0,1])
    lights = []
    objects = []
    return camera, lights, objects
