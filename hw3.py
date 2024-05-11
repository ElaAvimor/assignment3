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
            min_distance, nearest_object = ray.nearest_intersected_object(objects)
            get_color(ray, nearest_object, ambient, lights, intersection_point)

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
    if level == 0:
        return color
    else:

def calc_diffuse_color(ray, light, nearest_object):
    material_diffuse = nearest_object.diffuse
    intersection_normal =
    intersection_to_light =
    light_intensity = light.get_intensity(nearest_object)
    return material_diffuse * light_intensity * (np.dot( intersection_normal,intersection_to_light))

def calc_specular_color(ray, light, nearest_object):
    material_specular = nearest_object.specular
    intersection_to_eye =
    intersection_to_reflected_light =
    light_intensity = light.get_intensity(nearest_object)
    return  material_specular * light_intensity * ((np.dot(intersection_to_eye,intesection_to_reflected_light))**nearest_object.shininess)


def return_if_light_shadow(light, ray, p ,nearest_object):
    light_ray = light.get_light_ray(p)






# Write your own objects and lights
# TODO
def your_own_scene():
    camera = np.array([0,0,1])
    lights = []
    objects = []
    return camera, lights, objects
