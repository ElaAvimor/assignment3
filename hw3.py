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

            # This is the main loop where each pixel color is computed.
            # done by us
            min_distance, nearest_object, p = ray.nearest_intersected_object(objects)
            if nearest_object:
                color = get_color(ray, nearest_object, ambient, lights, p, max_depth, objects)

            
            # We clip the values between 0 and 1 so all pixel values will make sense.
            image[i, j] = np.clip(color,0,1)

    return image


def get_color(ray, nearest_object, ambient, lights, p, level, objects):
    global_ambient = np.array(ambient)
    material_ambient = np.array(nearest_object.ambient)
    ambient_color = global_ambient * material_ambient
    diffuse_color = 0
    specular_color = 0
    reflective_color = 0

    for light in lights:
        shadow = return_if_light_shadow(light, ray, p ,nearest_object, objects)
        if shadow == 1:
            diffuse_color = calc_diffuse_color(light, nearest_object, p)
            specular_color = calc_specular_color(ray, light, nearest_object, p)

    level -= 1
    if level > 0:
        reflected_ray = construct_reflective_ray(ray, p, nearest_object)
        min_distance, reflected_obj, reflected_point = reflected_ray.nearest_intersected_object(objects)
        if reflected_obj:
            reflective_color = reflected_obj.reflection * get_color(reflected_ray, reflected_obj, ambient, lights,reflected_point, level, objects)

    color = ambient_color + diffuse_color + specular_color + reflective_color
    return color.astype(np.float64)

def calc_diffuse_color(light, nearest_object, p):
    if isinstance(nearest_object, Sphere):
        intersection_normal = normalize(p - nearest_object.center)
    else:
        intersection_normal = normalize(nearest_object.normal)

    material_diffuse = nearest_object.diffuse
    intersection_to_light = light.get_light_ray(p).direction
    light_intensity = light.get_intensity(p)

    return material_diffuse * light_intensity * (np.dot(intersection_normal,intersection_to_light))

def calc_specular_color(ray, light, nearest_object, p):
    if isinstance(nearest_object, Sphere):
        normal = normalize(p - nearest_object.center)
    else:
        normal = normalize(nearest_object.normal)

    intersection_to_light = light.get_light_ray(p).direction
    material_specular = nearest_object.specular
    intersection_to_eye = -1 * ray.direction
    intersection_to_reflected_light = normalize(reflected(-1*intersection_to_light, normal))
    light_intensity = light.get_intensity(p)

    return  material_specular * light_intensity * ((np.dot(intersection_to_eye,intersection_to_reflected_light))**nearest_object.shininess)


def return_if_light_shadow(light, ray, p ,nearest_object, objects):
    light_ray = light.get_light_ray(p)
    min_distance, shadow_obj, light_intersection_point= light_ray.nearest_intersected_object(objects)

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

    # done by us:

    directional_light = DirectionalLight(intensity= np.array([0.7, 0.7, 0.7]),direction=np.array([0, 0, 1]))
    spotlight_a = SpotLight(np.array([1, 5, 0]), np.array([-2, -0.25, -2]), np.array([0, 1, -1]),0.1, kl=0.05, kq=0.05)
    spotlight_b = SpotLight(intensity=np.array([3, 0, 3]), position=np.array([1, -0.25, 1]), direction=np.array([1, 1, -1]), kc=0.1, kl=0.05, kq=0.05)

    lights = [directional_light, spotlight_a, spotlight_b]

    sphere = Sphere([0, 1, -1], 0.3)
    sphere.set_material([0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.5, 0.5, 0.5], 750, 1)

    background = Plane([0, 0, 1], [0, 0, -3])
    background.set_material([0, 0, 0], [0.5, 0.1, 0.5], [0.1, 0.1, 0.1], 0, 0)

    objects = [sphere, background]

    n_tiles = 4
    for i in range(-n_tiles, n_tiles):
        for j in range(-n_tiles, n_tiles):
            # First triangle of the square tile
            triangle1 = Triangle([i, -0.5, j], [i, -0.5, j + 1], [i + 1, -0.5, j + 1])
            # Second triangle of the square tile
            triangle2 = Triangle([i, -0.5, j], [i + 1, -0.5, j + 1], [i + 1, -0.5, j])

            # Set materials based on a checker pattern
            if (i + j) % 2 == 0:
                triangle1.set_material([1, 0, 0], [1, 0, 0], [0.2, 0.2, 0.2], 50, 0.1)
                triangle2.set_material([1, 0, 0], [1, 0, 0], [0.2, 0.2, 0.2], 50, 0.1)
            else:
                triangle1.set_material([0, 0, 1], [0, 0, 1], [0.2, 0.2, 0.2], 50, 0.1)
                triangle2.set_material([0, 0, 1], [0, 0, 1], [0.2, 0.2, 0.2], 50, 0.1)

            objects.append(triangle1)
            objects.append(triangle2)


    return camera, lights, objects
