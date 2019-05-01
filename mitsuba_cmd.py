# coding: utf-8

from xml.etree.ElementTree import Element, tostring
from xml.etree import ElementTree
from xml.dom import minidom
import subprocess
import numpy as np

def arr2str(arr):
    return ", ".join([str(a) for a in arr])

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
    def __init__(self, rgb=(1, 1, 1)):
        name = "spectrumProperty"
        values = [str(v) for v in rgb]
        str_values = ", ".join(values)
        super().__init__("rgb", name, str_values)    


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
    def __init__(self, value):
        """only uniform scale is supported"""
        super().__init__("scale")
        self.set("value", str(value))

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


class StradifiedSampler(BaseSamplerD):
    def __init__(self, sample_count=None, dimension=None):
        super().__init__("stratified", sample_count, dimension)

class LDSampler(BaseSamplerD):
    def __init__(self, sample_count=None, dimension=None):
        super().__init__("ldsampler", sample_count, dimension)


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


class BaseSensor(BaseElement):
    def __init__(self, _type, sampler, film):
        super().__init__("sensor")
        self.set("type", _type)
        self.extend([sampler, film])
        

class PerspectiveSensor(BaseSensor):
    def __init__(self, sampler, film):
        super().__init__("perspective", sampler, film)


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
    def __init__(self, fields):
        """
        
        Arguments:
            fields {list} -- a list of integrators
        """
        super().__init__("multichannel")
        self.extend(fields)

class PathIntegrator(BaseIntegrator):
    def __init__(self, max_depth=1, rr_depth=5, strict_normals=True, hide_emitters=False):
        super().__init__("path")
        max_d = Integer("maxDepth", max_depth)
        rr_d = Integer("rrDepth", rr_depth)
        strict_n = Boolean("strictNormals", strict_normals)
        hide_e = Boolean("hideEmitters", hide_emitters)

        self.extend([max_d, rr_d, strict_n, hide_e])
    



def run_mitsuba(xml_fname, output_fname=None, quiet=False):

    cmd = ["mitsuba"]
    if output_fname is not None:
        cmd += [f"-o {output_fname}"]
    if quiet:
        cmd += ["-q"]

    cmd += [str(xml_fname)]

    out = subprocess.run(cmd)
    return out


def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ElementTree.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

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
    sensor = PerspectiveSensor(sampler, film)
    
    lookat =  Lookat(origin=[0, 0, 5], target=[0, 0, 0], up=[0, 1, 0])
    sensor_transform = Transform("toWorld", [lookat])
    sensor.extend([sensor_transform])


    scene.extend([multi_integrator, sensor, sphere, disk])


    from pathlib import Path
    fname_out = Path("test.xml")
    save_cmd(fname_out, scene)

    run_mitsuba(fname_out, quiet=False)

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

