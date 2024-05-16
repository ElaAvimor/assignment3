import numpy as np


# This function gets a vector and returns its normalized form.
def normalize(vector):
    return vector / np.linalg.norm(vector)


# This function gets a vector and the normal of the surface it hit
# This function returns the vector that reflects from the surface
def reflected(vector, axis):
    # done by us
    n_axis = normalize(axis)
    return vector - 2 * np.dot(vector, n_axis) * n_axis

## Lights
class LightSource:
    def __init__(self, intensity):
        self.intensity = intensity


class DirectionalLight(LightSource):

    def __init__(self, intensity, direction):
        super().__init__(intensity)
        # done by us
        #self.direction = normalize(np.array(direction))
        self.direction = normalize(direction)

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self,intersection_point):
        # done by us
        #return Ray(intersection_point, -self.direction)
        n_direction = -normalize(self.direction)
        return Ray(intersection_point, n_direction)

    # This function returns the distance from a point to the light source
    def get_distance_from_light(self, intersection):
        # done by us
        return np.inf

    # This function returns the light intensity at a point
    def get_intensity(self, intersection):
        # done by us
        return self.intensity 


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
        #self.position = np.array(position)
        self.position = position
        #self.direction = normalize(np.array(direction))
        self.direction = np.array(direction)
        self.kc = kc
        self.kl = kl
        self.kq = kq

    # This function returns the ray that goes from the light source to a point
    def get_light_ray(self, intersection):
        # done by us
        return Ray(intersection, normalize(-self.direction))

    def get_distance_from_light(self, intersection):
        # done by us
        return np.linalg.norm(self.position - intersection)

    def get_intensity(self, intersection):
        direction_to_point = normalize(intersection - self.position)
        dot_product = np.dot(direction_to_point, self.direction)
        # Implement a smoother falloff curve
        effective_intensity = self.intensity * (max(dot_product, 0) ** 2)        
        distance = np.linalg.norm(intersection - self.position)
        attenuation = self.kc + self.kl * distance + self.kq * (distance ** 2)
        return effective_intensity / attenuation

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
        min_t = np.inf
        # done by us

        for obj in objects:
            obj_ray_intersection = obj.intersect(self)
            if obj_ray_intersection:
                t, current_obj = obj_ray_intersection
                if min_t > t: 
                    min_t = t
                    nearest_object = current_obj 
        if nearest_object is not None:
            p = self.origin + min_t * self.direction
            min_distance = np.linalg.norm(p-self.origin)
            return min_distance, nearest_object, p
        return min_distance, nearest_object, None


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
        denominator = np.dot(self.normal, ray.direction)

        if abs(denominator) > 1e-6:
            v = self.point - ray.origin

            t = np.dot(v, self.normal) / denominator

            if t > 0:
                return t, self

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
        super().__init__()
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
        edge1 = self.b - self.a
        edge2 = self.c - self.a
        h = np.cross(ray.direction, edge2)
        a = np.dot(edge1, h)
        if -1e-6 < a < 1e-6:
            return None  # The ray is parallel to this triangle
        f = 1.0 / a
        s = ray.origin - self.a
        u = f * np.dot(s, h)
        if u < 0.0 or u > 1.0:
            return None
        q = np.cross(s, edge1)
        v = f * np.dot(ray.direction, q)
        if v < 0.0 or u + v > 1.0:
            return None
        t = f * np.dot(edge2, q)
        if t > 1e-6:  # t must be positive and sufficiently large to be considered an intersection
            return t, self
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
            triangle = Triangle(self.v_list[idx[0]], self.v_list[idx[1]], self.v_list[idx[2]])
            l.append(triangle)
        return l

    def apply_materials_to_triangles(self):
        # done by us
        for triangle in self.triangle_list:
            triangle.set_material(self.ambient, self.diffuse, self.specular, self.shininess, self.reflection)

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
        self.center = np.array(center)
        self.radius = radius

    def intersect(self, ray: Ray):
        # construct a vector from the origin of the ray to the center of the sphere
        v = self.center - ray.origin
        # find the length of the projection of v on the ray 
        v_proj_len = np.dot(v, ray.direction)
        # if the length of the projection is negative, the ray points to the opposite direction to the sphere -> there isn't an intersection
        if v_proj_len < 0:
            return None
        
        # find the point on the ray which is the closest to the center of the sphere
        p = ray.origin + v_proj_len*ray.direction
        # calculate the distance from the center to p (which is the distane from the center to the ray)
        distance = np.linalg.norm(p - self.center)

        if distance > self.radius:
            # no intersection
            return None

        # calculate the length of the ray from the intersection point to p (using Pythagoras)
        ray_out_len = np.sqrt(self.radius**2 - distance**2)
        t = v_proj_len - ray_out_len
        return t, self

# Helper function to calculate t
# done by us
def quadratic_formula(A, B, C):
    if B**2 - 4*A*C >=0:
        discriminant = np.sqrt(B**2 - 4*A*C)
        if discriminant > 0:
            t1 = (-1*B + discriminant) / 2*A
            t2 = (-1*B - discriminant) / 2*A
            if t1 < 0 and t2 < 0:
                return None
            else:
                t = min(t1, t2)
                return t
        elif discriminant == 0:
            t1 = (-1*B + discriminant) / 2*A
            if t1 < 0:
                    return None
            else:
                t = min(t1, t2)
                return t
        else:
            return None
    else:
        return None

