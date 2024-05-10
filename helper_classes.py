import numpy as np


# This function gets a vector and returns its normalized form.
def normalize(vector):
    return vector / np.linalg.norm(vector)


# This function gets a vector and the normal of the surface it hit
# This function returns the vector that reflects from the surface
def reflected(vector, axis):
    # done by us
    axis = normalize(axis)
    return vector - 2 * np.dot(vector, axis) * axis

## Lights
class LightSource:
    def __init__(self, intensity):
        self.intensity = intensity


class DirectionalLight(LightSource):

    def __init__(self, intensity, direction):
        super().__init__(intensity)
        # done by us
        self.direction = normalize(np.array(direction))

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self,intersection_point):
        # done by us
        return Ray(intersection_point, -self.direction)

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self, intersection):
        # done by us
        return np.inf # Distance is considered infinite for directional light

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        # done by us
        return self.intensity # Intensity is the same for all points in the scene in directional light 


class PointLight(LightSource):
    def __init__(self, intensity, position, kc, kl, kq):
        super().__init__(intensity)
        self.position = np.array(position)
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from a point to the light source
    def get_light_ray(self,intersection):
        return Ray(intersection, normalize(self.position - intersection))

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self,intersection):
        return np.linalg.norm(intersection - self.position)

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        d = self.get_distance_from_light(intersection)
        return self.intensity / (self.kc + self.kl*d + self.kq * (d**2))


class SpotLight(LightSource):
    def __init__(self, intensity, position, direction, kc, kl, kq):
        super().__init__(intensity)
        # done by us
        self.position = np.array(position)
        self.direction = normalize(np.array(direction))
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self, intersection):
        # done by us
        return Ray(intersection, normalize(self.position - intersection))

    def get_distance_from_light(self, intersection):
        # done by us
        return np.linalg.norm(self.position - intersection)

    def get_intensity(self, intersection):
        # done by us
        d = self.get_distance_from_light(intersection)
        return self.intensity / (self.kc + self.kl*d + self.kq * (d**2))


class Ray:
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction 

    # The function is getting the collection of objects in the scene and looks for the one with minimum distance.
    # The function should return the nearest object and its distance (in two different arguments)
    def nearest_intersected_object(self, objects):
        intersections = None
        nearest_object = None
        min_distance = np.inf
        # done by us

        for obj in objects:
            obj_ray_intersection = obj.intersect(self)
            if obj_ray_intersection:
                t, current_obj = obj_ray_intersection
                if min_distance > t: 
                    min_distance = t
                    nearest_object = current_obj 

        return nearest_object, min_distance


class Object3D:
    def set_material(self, ambient, diffuse, specular, shininess, reflection):
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        self.shininess = shininess
        self.reflection = reflection


class Plane(Object3D):
    def __init__(self, normal, point):
        self.normal = np.array(normal)
        self.point = np.array(point)

    def intersect(self, ray: Ray):
        v = self.point - ray.origin
        t = np.dot(v, self.normal) / (np.dot(self.normal, ray.direction) + 1e-6)
        if t > 0:
            return t, self
        else:
            return None


class Triangle(Object3D):
    """
        C
        /\*
       /  \*
    A /____\ B

    The fornt face of the triangle is A -> B -> C.
    
    """
    def __init__(self, a, b, c):
        self.a = np.array(a)
        self.b = np.array(b)
        self.c = np.array(c)
        self.normal = self.compute_normal()
        self.plane = Plane(self.normal, self.a)

    # computes normal to the trainagle surface. Pay attention to its direction!
    def compute_normal(self):
        # done by us
        edge_ba = self.b - self.a
        edge_ca = self.c - self.a
        return normalize(np.cross(edge_ba, edge_ca))

    def intersect(self, ray: Ray):
        # done by us
        intPlane = self.plane.intersect(ray)
        if intPlane is not None:
            t, plane = intPlane
            P = ray.origin + t*ray.direction
            edge_ba = self.b - self.a
            edge_ca = self.c - self.a
            triangle_area = np.linalg.norm(np.cross(edge_ba,edge_ca)) /2
            alpha = np.linalg.norm(np.cross((self.b - P), (self.c - P))/ 2 * triangle_area)
            beta = np.linalg.norm(np.cross((self.b - P), (self.c - P))/ 2 * triangle_area)
            gamma = 1 - alpha - beta
            if (0 <= alpha <= 1) and (0 <= beta <= 1) and (0 <= gamma <= 1) and (alpha + beta + gamma == 1):
                return (t, self)
            else:
                return None
        else:
            return None
        

class Pyramid(Object3D):
    """     
            D
            /\*\*
           /==\**\*
         /======\***\*
       /==========\***\*
     /==============\****\*
   /==================\*****\*
A /&&&&&&&&&&&&&&&&&&&&\ B &&&/ C
   \==================/****/
     \==============/****/
       \==========/****/
         \======/***/
           \==/**/
            \/*/
             E 
    
    Similar to Traingle, every from face of the diamond's faces are:
        A -> B -> D
        B -> C -> D
        A -> C -> D
        E -> B -> A
        E -> C -> B
        C -> E -> A
    """
    def __init__(self, v_list):
        self.v_list = v_list
        self.triangle_list = self.create_triangle_list()

    def create_triangle_list(self):
        l = []
        t_idx = [
                [0,1,3],
                [1,2,3],
                [0,3,2],
                 [4,1,0],
                 [4,2,1],
                 [2,4,0]]
        for idx in t_idx:
            triangle = Triangle.__init__(idx)
            l.append(triangle)
        return l

    def apply_materials_to_triangles(self):
        # done by us
        for triangle in self.triangle_list:
            triangle.set_material(self.ambient, self.defuse, self.specular, self.shininess, self.reflection)

    def intersect(self, ray: Ray):
        # done by us
        minimal_t = np.inf
        nearest_triangle = None

        for triangle in self.triangle_list:
            if triangle.intersect(ray) is not None:
                t, current_triangle = triangle.intersect(ray)

                if t < minimal_t:
                    nearest_triangle = current_triangle
                    minimal_t = t
        if not nearest_triangle:
            return None
        return minimal_t, nearest_triangle


class Sphere(Object3D):
    def __init__(self, center, radius: float):
        self.center = center
        self.radius = radius

    def intersect(self, ray: Ray):
        # done by us
        A_coefficient = np.dot(ray.direction)
        center_to_origin_vector = ray.origin - self.center
        B_coefficient = 2 * np.dot(ray.direction, center_to_origin_vector)
        C_coefficient = np.dot(center_to_origin_vector) - self.radius**2

        t = quadratic_formula(self, A_coefficient, B_coefficient, C_coefficient)
        if t is not None:
            return t, self
        else:
            return None

# Helper function to calculate t
# done by us
def quadratic_formula(A, B, C):
        discriminant = np.sqrt(B**2 - 4*A*C)
        if discriminant > 0:
            t1 = (-1*B + discriminant) / 2*A
            t2 = (-1*B - discriminant) / 2*A
            if t1 < 0 and t2 < 0:
                return None
            else:
                t = np.min(t1, t2)
                return t
        else:
            return None

