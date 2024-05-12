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
        if shadow == 0:
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
    camera = np.array([0, 0, 1])
    
    # Lights
    directional_light = DirectionalLight(intensity= np.array([0.7, 0.7, 0.7]),direction=np.array([0, 0, 4]))
    
    spotlight_a = SpotLight(intensity=np.array([1, 5, 0]), position=np.array([-2, -0.25, -2]), direction=np.array([0, 1, -1]),
                         kc=0.1, kl=0.05, kq=0.05)
    
    spotlight_b = SpotLight(intensity=np.array([3, 0, 3]), position=np.array([1, -0.25, 1]), direction=np.array([1, 1, -1]),
                         kc=0.1, kl=0.05, kq=0.05)

    lights = [directional_light, spotlight_a, spotlight_b]
    
    # Objects
    plane = Plane(normal=[0, 1, 0], point=[0, -1, 0])
    sphere = Sphere(center=[0, 0, 3], radius=1)
    pyramid = Pyramid(v_list=[[-1, 0, 2], [1, 0, 2], [0, 0, 4], [0, 2, 3], [0, -2, 3]])
    
    pyramid.set_material(
        ambient=[0.2, 0.2, 0.2], 
        diffuse=[0.2, 0.2, 0.2], 
        specular=[0.5, 0.5, 0.5], 
        shininess=750, 
        reflection=0.5
    )

    pyramid.apply_materials_to_triangles()
    
    sphere.set_material(
        ambient=[0.2, 0.2, 0.2], 
        diffuse=[1, 0, 0], 
        specular=[1, 1, 1], 
        shininess=100, 
        reflection=0.3
    )
    plane.set_material(
        ambient=[0.2, 0.2, 0.2], 
        diffuse=[0.3, 0.4, 0.5], 
        specular=[0.2, 0.2, 0.2], 
        shininess=10, 
        reflection=0.1
    )

    objects = [plane, sphere, pyramid]
    
    return camera, lights, objects