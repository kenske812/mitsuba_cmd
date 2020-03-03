# coding: utf-8

from xml.etree.ElementTree import Element, tostring
from xml.etree import ElementTree
from xml.dom import minidom
import subprocess
import numpy as np

def arr2str(arr):
    return ", ".join([str(a) for a in arr])

def check_numeric(v):
    if isinstance(v, int) or isinstance(v, float):
        return True
    else:
        return False

class BaseElement(Element):
    def __init__(self, tag):
        super().__init__(tag)

    def extend(self, elements):

        if not (isinstance(elements, list) or isinstance(elements, Element)):
                raise TypeError(f"input should be a list or {type(Element)}, not {type(elements)}")

        if isinstance(elements, Element):
            super().extend([elements])
        else:            
            super().extend(elements)
    
    def change_property_value(self, name, value):
        """change value if "name" property is name.
        
        Arguments:
            name {[type]} -- name to be changed 
            value {[type]} -- value to replace
        
        Raises:
            TypeError: [description]
        """
        for e in self.iter():
            if "name" in e.attrib:
                if e.attrib["name"] == name:
                    t1 = type(e.attrib["value"])
                    t2 = type(value)
                    if t1 != t2:
                        raise TypeError(f"type of the input value should be {t1} but got {t2}")
                    
                    e.attrib["value"] = value 
    


class Scene(BaseElement):
    def __init__(self, version="0.5.0"):
        super().__init__("scene")
        self.set("version", version)


class BaseProperty(BaseElement):
    def __init__(self, tag, name, value):
        super().__init__(tag)
        self.set("name", name)
        self.set("value", str(value))

class Boolean(BaseProperty):
    def __init__(self, name, value):
        if value == True:
            v = "true"
        else:
            v = "false"
        super().__init__("boolean", name, v)

class Float(BaseProperty):
    def __init__(self, name, value):
        super().__init__("float", name, value)

class Integer(BaseProperty):
    def __init__(self, name, value):
        super().__init__("integer", name, value)    

class String(BaseProperty):
    def __init__(self, name, value):
        super().__init__("string", name, value)    


class RGB(BaseProperty):
    def __init__(self, name, rgb=(1, 1, 1)):
        values = [str(v) for v in rgb]
        str_values = ", ".join(values)
        super().__init__("rgb", name, str_values)    

class Spectrum(BaseProperty):
    def __init__(self, name, values):
        if check_numeric(values):
            str_values = str(values)
        else:
            str_values = arr2str(values)
        super().__init__("spectrum", name, str_values)


class Translate(BaseElement):
    def __init__(self, xyz=(1,1,1)):
        super().__init__("translate")
        self.set("x", str(xyz[0]))
        self.set("y", str(xyz[1]))
        self.set("z", str(xyz[2]))

class Rotate(BaseElement):
    def __init__(self, xyz=(1,1,1), angle="180"):
        super().__init__("rotate")
        self.set("x", str(xyz[0]))
        self.set("y", str(xyz[1]))
        self.set("z", str(xyz[2]))
        self.set("angle", str(angle))

class Scale(BaseElement):
    def __init__(self, x, y=None, z=None):
        super().__init__("scale")
        if y is None and z is None:
            self.set("value", str(x))
        else:
            self.set("x", str(x))
            if y is not None:
                self.set("y", str(y))
            if z is not None:
                self.set("z", str(z))
            

class Point(BaseElement):
    def __init__(self, name, xyz):
        super().__init__("point")
        self.set("name", name)
        self.set("x", str(xyz[0]))
        self.set("y", str(xyz[1]))
        self.set("z", str(xyz[2]))

class Vector(BaseElement):
    def __init__(self, name, xyz):
        super().__init__("vector")
        self.set("name", name)
        self.set("x", str(xyz[0]))
        self.set("y", str(xyz[1]))
        self.set("z", str(xyz[2]))    



class Lookat(BaseElement):
    def __init__(self, origin, target, up=None):
        super().__init__("lookat")
        self.set("origin", arr2str(origin))
        self.set("target", arr2str(target))
        if up is not None:
            self.set("up", arr2str(up))
        

class Transform(BaseElement):
    def __init__(self, name, transforms=[]):
        super().__init__("transform")
        self.set("name", name)
        self.extend(transforms)


class BaseSampler(BaseElement):
    def __init__(self, _type, sample_count=None):
        super().__init__("sampler")
        self.set("type", _type)
        
        if sample_count is not None:
            count = Integer("sampleCount", sample_count)
            self.extend([count])

class BaseSamplerD(BaseSampler):
    def __init__(self, _type, sample_count=None, dimension=None):
        super().__init__(_type, sample_count)
        if dimension is not None:
            d = Integer("dimension", dimension)
            self.extend([d])


class IndependentSampler(BaseSampler):
    def __init__(self, sample_count=None):
        super().__init__("independent", sample_count=sample_count)

class StradifiedSampler(BaseSamplerD):
    def __init__(self, sample_count=None, dimension=None):
        super().__init__("stratified", sample_count, dimension)

class LDSampler(BaseSamplerD):
    def __init__(self, sample_count=None, dimension=None):
        super().__init__("ldsampler", sample_count, dimension)

class HaltonSampler(BaseSampler):
    def __init__(self, sample_count=None, scramble=None):
        super().__init__("halton", sample_count=sample_count)
        if scramble is not None:
            self.append(Integer("scramble", scramble))
        
        

class BaseFilm(BaseElement):
    def __init__(self, _type, width, height):
        super().__init__("film")
        self.set("type", _type)

        banner = Boolean("banner", False)
        w = Integer("width", width)
        h = Integer("height", height)
        self.extend([banner, h, w])

class HDRFilm(BaseFilm):
    def __init__(self, width, height, pixel_formats=[], channel_names=[]):
        """
        Arguments:
            width {int} -- # of pixels of width
            height {int} -- # of pixel of height
        
        Keyword Arguments:
            multi_channel {bool} -- whether output normals and distance (default: {False})
        """
        super().__init__("hdrfilm", width, height)
        if len(pixel_formats) != 0:
            if len(pixel_formats) != len(channel_names):
                raise ValueError("lengths of pixel_formats and channel_names are inconsistent")

            fmt_value = ", ".join(pixel_formats)
            str_pixel_format = String("pixelFormat", fmt_value)

            ch_value = ", ".join(channel_names)
            str_channel_names = String("channelNames", ch_value)

            self.extend([str_pixel_format, str_channel_names])

class BaseShape(BaseElement):
    def __init__(self, _type):
        super().__init__("shape")
        self.set("type", _type)


class Sphere(BaseShape):
    def __init__(self, radius):
        super().__init__("sphere")
        r = Float("radius", radius)
        self.extend([r])

class Disk(BaseShape):
    def __init__(self, radius):
        super().__init__("disk")
        r = Transform("toWorld", [Scale(radius)])
        self.extend(r)

class PLY(BaseShape):
    def __init__(self, filename, to_world=None, 
                       face_normals=None, max_smooth_angle=None, 
                       flip_normals=None, srgb=None):
        super().__init__("ply") 
        self.extend(String("filename", str(filename)))

        if to_world is not None:
            self.extend(to_world)
        if face_normals is not None:
            self.extend(Boolean("faceNormals", face_normals))
        if max_smooth_angle is not None:
            self.extend(Float("maxSmoothAngle", max_smooth_angle))
        if flip_normals is not None:
            self.extend(Boolean("flipNormals", flip_normals))
        if srgb is not None:
            self.extend(Boolean("srgb", srgb))

class BaseTexture(BaseElement):
    def __init__(self, _type):
        super().__init__("texture")
        self.set("type", _type)

class Bitmap(BaseTexture):
    def __init__(self, filename):
        super().__init__("bitmap")
        self.append(String("filename", filename))

class VertexColor(BaseTexture):
    def __init__(self):
        super().__init__("vertexcolors")
        self.set("name", "reflectance")


class BaseBSDF(BaseElement):
    def __init__(self, _type):
        super().__init__("bsdf")
        self.set("type", _type)

class Diffuse(BaseBSDF):
    def __init__(self, reflectance):
        """        
        Arguments:
            reflectance {float or list} -- [description]
        """
        super().__init__("diffuse")

        self.extend(Spectrum("reflectance", reflectance))


class DiffuseTexture(BaseBSDF):
    def __init__(self, texture):
        """        
        Arguments:
            reflectance {float or list} -- [description]
        """
        super().__init__("diffuse")

        self.append(texture)


class SmoothConductor(BaseBSDF):
    def __init__(self, material=None):
        super().__init__("conductor")
        if material is not None:
            self.add(String("material", material))
        


class RoughConductor(BaseBSDF):
    def __init__(self, distribution=None, alpha=None, material=None, 
                        eta=None, k=None, ext_eta=None):
        """
        
        Keyword Arguments:
            distribution {str} -- ["beckmann", "ggx", "phong"] (default: {None})
            alpha {float} -- roughness (default: {None})
            material {str} -- [description] (default: {None})
            eta {float or list} -- [description] (default: {None})
            k {float or list} -- [description] (default: {None})
            ext_eta {float} -- [description] (default: {None})
        """
        super().__init__("roughconductor")

        if distribution is not None:
            self.append(String("distribution", distribution))

        if alpha is not None:
            self.append(Float("alpha", alpha))
        if material is not None:
            self.append(String("material", material))
        if eta is not None:
            self.append(Spectrum("eta", eta))
        if k is not None:
            self.append(Spectrum("k", k))
        if ext_eta is not None:
            if isinstance(ext_eta, str):
                self.append(String("extEta", ext_eta))
            else:
                self.append(Float("extEta", ext_eta))

class SmoothPlastic(BaseBSDF):
    def __init__(self, int_ior=None, ext_ior=None, 
                       specular_reflectance=None,
                       diffuse_reflectance=None,
                       nonlinear=None):
        """[summary]
        
        Keyword Arguments:
            int_ior {float or str} --  Interior index of refraction. defulat ispolypropylene/ 1.49 (default: {None})
            ext_ior {float or str} -- Exterior index of refraction. defualt is air(default: {None})
            specular_reflectance {float or list} -- default is 1.0 (default: {None})
            diffuse_reflectance {float or list} -- default is 0.5 (default: {None})
            nonlinear {bool} -- if dispersion is acounted.(default: {None})
        """
        super().__init__("plastic")
        if int_ior is not None:
            if isinstance(int_ior, float):
                self.append(Float("intIOR", int_ior))
            elif isinstance(int_ior, str):
                self.append(String("intIOR", int_ior))
            else:
                TypeError(f"use float or str, not {type(int_ior)}")
        
        if ext_ior is not None:
            if isinstance(ext_ior, float):
                self.append(Float("extIOR", ext_ior))
            elif isinstance(ext_ior, str):
                self.append(String("extIOR", ext_ior))
            else:
                TypeError(f"use float or str, not {type(ext_ior)}")
        
        if specular_reflectance is not None:
            self.append(Spectrum("specularReflectance", specular_reflectance))
        if diffuse_reflectance is not None:
            self.append(Spectrum("diffuseReflectance", diffuse_reflectance))

        if nonlinear is not None:
            self.append(Boolean("nonlinear", nonlinear))


            
class RoughPlastic(BaseBSDF):
    def __init__(self, distribution=None, alpha=None, 
                       int_ior=None, ext_ior=None, 
                       specular_reflectance=None,
                       diffuse_reflectance=None,
                       nonlinear=None):
        """
        
        Keyword Arguments:
            distribution {str} -- ["beckmann", "ggx", "phong"] (default: {None})
            alpha {float} -- roughness (default: {None})
            int_ior {float or str} --  Interior index of refraction. defulat ispolypropylene/ 1.49 (default: {None})
            ext_ior {float or str} -- Exterior index of refraction. defualt is air(default: {None})
            specular_reflectance {float or list} -- default is 1.0 (default: {None})
            diffuse_reflectance {float or list} -- default is 0.5 (default: {None})
            nonlinear {bool} -- if dispersion is acounted.(default: {None})
        """     

        super().__init__("roughplastic")

        if distribution is not None:
            self.append(String("distribution", distribution))

        if alpha is not None:
            self.append(Float("alpha", alpha))

        if int_ior is not None:
            if isinstance(int_ior, float):
                self.append(Float("intIOR", int_ior))
            elif isinstance(int_ior, str):
                self.append(String("intIOR", int_ior))
            else:
                TypeError(f"use float or str, not {type(int_ior)}")
        
        if ext_ior is not None:
            if isinstance(ext_ior, float):
                self.append(Float("extIOR", ext_ior))
            elif isinstance(ext_ior, str):
                self.append(String("extIOR", ext_ior))
            else:
                TypeError(f"use float or str, not {type(ext_ior)}")
        
        if specular_reflectance is not None:
            self.append(Spectrum("specularReflectance", specular_reflectance))
        if diffuse_reflectance is not None:
            self.append(Spectrum("diffuseReflectance", diffuse_reflectance))

        if nonlinear is not None:
            self.append(Boolean("nonlinear", nonlinear))




class BaseSensor(BaseElement):
    def __init__(self, _type):
        super().__init__("sensor")
        self.set("type", _type)
        

class PerspectiveCamera(BaseSensor):
    def __init__(self, fov, fov_axis, origin, target, up):
        """perspective camera
        for fov_axis:
            (i) x: fov maps to the x-axis in screen space.
            (ii) y: fov maps to the y-axis in screen space.
            (iii) diagonal: fov maps to the screen diagonal.
            (iv) smaller: fov maps to the smaller dimension (e.g. x when width<height)
            (v) larger: fov maps to the larger dimension (e.g. y when width<height)
            The default is x.
        Arguments:
            BaseSensor {[type]} -- [description]
            fov {float} -- fov in degree(0-180)
            fov_axis {str} -- [description]
            origin {[type]} -- [description]
            target {[type]} -- [description]
            up {[type]} -- [description]
        """
        super().__init__("perspective")
        
        float_fov = Float("fov", fov)
        str_fov_axis = String("fovAxis", fov_axis)

        to_world = Transform("toWorld")
        lookat = Lookat(origin, target, up)
        to_world.extend(lookat)

        self.extend([to_world, float_fov, str_fov_axis])




class OrthographicCamera(BaseSensor):
    def __init__(self, scale_x, scale_y, origin, target, up):
        super().__init__("orthographic")
        to_world = Transform("toWorld")
        scale = Scale(scale_x, scale_y)
        lookat = Lookat(origin, target, up)
        to_world.extend([scale, lookat])
        self.extend(to_world)
    


class BaseEmitter(BaseElement):
    def __init__(self, _type):
        super().__init__("emitter")
        self.set("type", _type)
    

class PointSource(BaseEmitter):
    def __init__(self, xyz, intensity_value, sampling_weight=None):
        """[summary]
        
        Arguments:
            xyz {list or ndarray} -- [x, y, z]
            intensity_value {float or list} -- spectrum intensity
        
        Keyword Arguments:
            sampling_weight {[type]} -- [description] (default: {None})
        """
        super().__init__("point")
        pos = Point("position", xyz)
        inten = Spectrum("intensity", intensity_value)

        self.extend([pos, inten])

        if sampling_weight is not None:
            float_sw = Float("samplingWeight", sampling_weight)
            self.extend(float_sw)

class SpotLightSource(BaseEmitter):
    def __init__(self, origin, target, intensity, cutoff_angle, beam_width=None):
        super().__init__("spot")
        to_world = Transform("toWorld")
        to_world.append(Lookat(origin, target))
        self.append(to_world)

        inten = Spectrum("intensity", intensity)
        self.append(inten)

        cutoff = Float("cutoffAngle", cutoff_angle)
        self.append(cutoff)

        if beam_width is not None:
            beam = Float("beamWidth", beam_width)
            self.append(beam)





class DirectionalSource(BaseEmitter):
    def __init__(self, direction, irradiance_value, sampling_weight=None):
        """
        Arguments:
            direction {list or ndarray} -- direction vector[x, y, z]
            irradiance_value {float or list} -- spectrum irradiance
        
        Keyword Arguments:
            sampling_weight {[type]} -- [description] (default: {None})
        """
        super().__init__("directional")
        vec = Vector("direction", direction)

        irr = Spectrum("irradiance", irradiance_value)


        self.extend([vec, irr])

        if sampling_weight is not None:
            float_sw = Float("samplingWeight", sampling_weight)
            self.extend(float_sw)

class OrthographicProjector(BaseEmitter):
    def __init__(self, fname, scale_x, scale_y, origin, target, up, irradiance_value):
        super().__init__("orthographicprojector")
        
        str_fname = String("filename", str(fname))

        if check_numeric(irradiance_value):
            irr = Float("irradiance", irradiance_value)
        else:
            irr = Spectrum("irradiance", irradiance_value)

        scale = Scale(scale_x, scale_y)
        lookat = Lookat(origin, target, up)
        trans = Transform("toWorld", [scale, lookat])

        self.extend([str_fname, trans, irr])

class PerspectiveProjector(BaseEmitter):
    def __init__(self, fname, fov, fov_axis, origin, target, up, radiance_scale):
        super().__init__("perspectiveprojector")
        str_fname = String("filename", str(fname))
        float_scale = Float("scale", radiance_scale)

        float_fov = Float("fov", fov)
        str_fov_axis = String("fovAxis", fov_axis)
        
        lookat = Lookat(origin, target, up)
        trans = Transform("toWorld", [lookat])

        self.extend([str_fname, float_scale, float_fov, str_fov_axis, trans])

class BaseIntegrator(BaseElement):
    def __init__(self, _type):
        super().__init__("integrator")
        self.set("type", _type)

class FieldExtractionIntegrator(BaseIntegrator):
    def __init__(self, value):
        """[summary]
        
        Arguments:
            value {string} -- e.g. "shNormal", "distance", "position". 
                              see the document(8.10.18) for further detail.
        """
        super().__init__("field")
        field = String("field", value)
        self.extend(field)

class MultiChannelIntegrator(BaseIntegrator):
    def __init__(self, integrators):
        """
        
        Arguments:
            integrators {list} -- a list of integrators
        """
        super().__init__("multichannel")
        self.extend(integrators)
    
    @classmethod
    def from_field_names(cls, fields, base_integrator=None):
        """[summary]
        
        Arguments:
            base_integrator {BaseIntegrator} -- base integrator to be used for rendering such as PathIntegrator.
            fields {list of str} -- each string should specify "shNormal", "distance" or "position". 
                                   see the documentation(8.10.18) for further detail.
        """
        if base_integrator is None:
            integrators = []
        else:
            integrators = [base_integrator]
            
        integrators += [FieldExtractionIntegrator(field) for field in fields]

        return cls(integrators)

class PathTracer(BaseIntegrator):
    def __init__(self, max_depth=2, rr_depth=5, strict_normals=True, hide_emitters=False):
        super().__init__("path")
        max_d = Integer("maxDepth", max_depth)
        rr_d = Integer("rrDepth", rr_depth)
        strict_n = Boolean("strictNormals", strict_normals)
        hide_e = Boolean("hideEmitters", hide_emitters)

        self.extend([max_d, rr_d, strict_n, hide_e])

class BidirectionalPT(BaseIntegrator):
    def __init__(self, max_depth=2, rr_depth=5, light_image=False, sample_direct=True):
        super().__init__("bdpt")
        max_d = Integer("maxDepth", max_depth)
        rr_d = Integer("rrDepth", rr_depth)
        l_image = Boolean("lightImage", light_image)
        sd = Boolean("sampleDirect", sample_direct)

        self.extend([max_d, rr_d, l_image, sd])

class PhotonMapper(BaseIntegrator):
    def __init__(self, max_depth=None, rr_depth=None, global_photons=None):
        super().__init__("photonmapper")
        if max_depth is not None:
            self.append(Integer("maxDepth", max_depth))
        if rr_depth is not None:
            self.append(Integer("rrDepth", rr_depth))
        if global_photons is not None:
            self.append(Integer("globalPhotons", global_photons))
            

class SPPM(BaseIntegrator):
    def __init__(self, max_depth=None, rr_depth=None, max_passes=None):
        super().__init__("sppm")
        if max_depth is not None:
            self.append(Integer("maxDepth", max_depth))
        if rr_depth is not None:
            self.append(Integer("rrDepth", rr_depth))
        if max_passes is not None:
            self.append(Integer("maxPasses", max_passes))

class PSSMLT(BaseIntegrator):
    def __init__(self, max_depth=None, rr_depth=None):
        super().__init__("pssmlt")
        if max_depth is not None:
            self.append(Integer("maxDepth", max_depth))
        if rr_depth is not None:
            self.append(Integer("rrDepth", rr_depth))
        





class ERPT(BaseIntegrator):
    def __init__(self, max_depth=None, direct_samples=None, 
                       lens_perturbation=None, multichain_perturbation=None):
        """camera needs to be perspective
        
        Keyword Arguments:
            max_depth {[type]} -- [description] (default: {None})
            direct_samples {[type]} -- [description] (default: {None})
            lens_perturbation {[type]} -- [description] (default: {None})
            multichain_perturbation {[type]} -- [description] (default: {None})
        """
        super().__init__("erpt")
        if max_depth is not None:
            self.append(Integer("maxDepth", max_depth))
        if direct_samples is not None:
            self.append(Integer("directSamples", direct_samples))
        if lens_perturbation is not None:
            self.append(Boolean("lensPerturbation", lens_perturbation))
        if multichain_perturbation is not None:
            self.append(Boolean("multiChainPertubation", multichain_perturbation))
        





class MitsubaError(Exception):
    pass

def run_mitsuba(xml_fname, output_fname=None, quiet=False):

    cmd = ["mitsuba"]
    if output_fname is not None:
        cmd += [f"-o{str(output_fname)}"]
    if quiet:
        cmd += ["-q"]

    cmd += [str(xml_fname)]

    out = subprocess.run(cmd)
    if out.returncode != 0:
        raise MitsubaError(f"{out}")
    return out


def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="    ")

def save_cmd(fname, elem):
    with open(fname, 'w') as f:
        f.write(prettify(elem))


def extract_multi_channels(exr_img, pixel_formats, channel_names):
    """extract (color, distance, normal) images from a EXR image
    
    Arguments:
        exr_img {} -- output of pyexr.open()
    
    Returns:
        [a list of ndarray] -- 
    """
    
    imgs = []
    for fmt, ch_name in zip(pixel_formats, channel_names):
        if fmt == "rgb":
            img = _extract_rgb_ch(exr_img, ch_name)
        elif fmt == "luminance":
            img = _extract_luminance_ch(exr_img, ch_name)

        else:
            raise NotImplementedError(f"only rgb and luminance format are supported, but got {fmt}.")
        
        imgs.append(img)

    return imgs

def _extract_rgb_ch(exr_img, ch_name):
    """extract rgb channel from EXR image
    
    Arguments:
        exr_img {EXR image} -- output of pyexr.open()
        ch_name {str} -- channel name
    
    Returns:
        [type] -- [description]
    """
    rgb = np.concatenate([exr_img.get(f"{ch_name}.{c}") for c in ["R", "G", "B"]], axis=-1)
    rgb = np.rollaxis(rgb, 2, 0) #(h, w, ch) -> (ch, h, w)
    return rgb

def _extract_luminance_ch(exr_img, ch_name):
    """extract Y channel from EXR image
    
    Arguments:
        exr_img {[type]} -- [description]
        ch_name {[type]} -- [description]
    
    Returns:
        [type] -- [description]
    """
    return exr_img.get(f"{ch_name}.Y")[:, :, 0]

def rgb_to_uint8(rgb):
    vmax = np.max(rgb)
    vmin = np.min(rgb)

    v = 255 * (rgb - vmin) / (vmax - vmin)
    v = v.astype(np.uint8)

    if v.shape[0] == 3:
        return np.rollaxis(v, 0, 3) # (3, h, w) -> (h, w, 3)
    else:
        return v
    


def normal_to_uint8(n):
    colored = n2color(n)
    return ncolor_to_uint8(colored)

def ncolor_to_uint8(ncolor):
    """color vector is computed by n2color"""

    v = (255*ncolor).astype(np.uint8)
    if v.shape[0] == 3:
        return np.rollaxis(v, 0, 3) #(3, h, w) -> (h, w, 3)
    else:
        return v

def n2color(n):
    return (n+1) / 2
    


if __name__ == "__main__":
    
    scene = Scene(version="0.5.0")
    sphere = Sphere(radius=0.5)
    disk = Disk(5)


    path_integrator = PathIntegrator(max_depth=1)
    normal_integrator = FieldExtractionIntegrator("shNormal")
    pos_integrator = FieldExtractionIntegrator("position")
    dist_integrator = FieldExtractionIntegrator("distance")
    multi_integrator = MultiChannelIntegrator([path_integrator, 
                                               normal_integrator, 
                                               pos_integrator,
                                               dist_integrator])

    pixel_formats = ["rgb", "rgb", "rgb", "luminance"]
    ch_names = ["color", "normal", "position", "distance"]
    
    sampler = LDSampler()
    film = HDRFilm(width=128, height=100, pixel_formats=pixel_formats, channel_names=ch_names)
    sensor = PerspectiveCamera(sampler, film, origin=[0, 0, 5], target=[0, 0, 0], up=[0, 1, 0])



    scene.extend([multi_integrator, sensor, sphere, disk])


    from pathlib import Path
    fname_out = Path("test.xml")
    save_cmd(fname_out, scene)

    run_mitsuba(fname_out, quiet=False)
    
    exit()

    import pyexr
    exr_img = pyexr.open(str(fname_out.with_suffix(".exr")))
    color, normal, position, dist = extract_multi_channels(exr_img, pixel_formats, ch_names)

    normal_uint8 = normal_to_uint8(normal)
    color_uint8 = rgb_to_uint8(color)
    pos_z = position[2]

    import matplotlib.pyplot as plt
    base_dir = Path(".")

    plt.title("color")
    plt.imshow(color_uint8)
    #plt.colorbar()
    plt.savefig(base_dir / "color.png")
    plt.clf()

    plt.title("distance")
    plt.imshow(dist)
    plt.colorbar()
    plt.savefig(base_dir / "distance.png")
    plt.clf()

    plt.title("normal")
    plt.imshow(normal_uint8)
    plt.savefig(base_dir / "normal.png")
    plt.clf()
    np.save("normal.npy", normal)

