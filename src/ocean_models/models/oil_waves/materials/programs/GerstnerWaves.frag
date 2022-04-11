// Copyright (c) 2016 The UUV Simulator Authors.
// All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Input parameters
uniform sampler2D bumpMap;
uniform samplerCube cubeMap;
uniform vec4 deepColor;
uniform vec4 shallowColor;
uniform float fresnelPower;
uniform float hdrMultiplier;

// Input computed in vertex shader
varying mat3 rotMatrix;
varying vec3 eyeVec;
varying vec2 bumpCoord;
varying vec4 pos;


vec4 wavelength_to_rgb(float wl)
{
    // https://en.wikipedia.org/wiki/Spectral_color
    // get color from light spectra
    vec3 color;
    if(wl < 380.0)
    {
        color = vec3(1.0, 0.0, 1.0);
    }
    else if(wl<440.0)
    {
        color = vec3(-(wl - 440.0) / (440.0 - 380.0), 0.0, 1.0);
    }
    else if(wl<490.0)
    {
        color = vec3(0.0, (wl - 440.0) / (490.0 - 440.0), 1.0);
    }
    else if(wl<510.0)
    {
        color = vec3(0.0, 1.0, -(wl - 510.0) / (510.0 - 490.0));
    }
    else if(wl<580.0)
    {
        color = vec3((wl - 510.0) / (580.0 - 510.0), 1.0, 0.0);
    }
    else if(wl<645.0)
    {
        color = vec3(1.0, -(wl - 645.0) / (645.0 - 580.0), 0.0);
    }
    else
    {
        color = vec3(1.0, 0.0, 0.0);
    }

    // scale to gamma
    float g = 0.80;
    vec3 gamma = vec3(g,g,g);
    color = pow(color, gamma);

    // set infra red and ultraviolet to 0 alpha
    float factor;
    if((wl >= 380.0) && (wl<420.0))
    {
        factor = (wl - 380.0) / (420.0 - 380.0);
    }
    else if((wl >= 420.0) && (wl<701.0))
    {
        factor = 1.0;
    }
    else if((wl >= 701.0) && (wl<781.0))
    {
        factor = (780.0 - wl) / (780.0 - 700.0);
    }
    else
    {
        factor = 0.0;
    }

    return vec4(color, factor);
}


vec4 frequency_to_rgb(float position)
{
  float f = mix(385.0, 785.0, position);

  float wavelen = 300000.0 / f;

  return wavelength_to_rgb(wavelen);
}

void main(void)
{
    // Apply bump mapping to normal vector to make waves look more detailed:
    vec4 bump = texture2D(bumpMap, bumpCoord)*2.0 - 1.0;
    vec3 N = normalize(rotMatrix * bump.xyz);

    // Reflected ray:
    vec3 E = normalize(eyeVec);
    vec3 R = reflect(E, N);
    // Gazebo requires rotated cube map lookup.
    R = vec3(R.x, R.z, R.y);

    // Get environment color of reflected ray:
    vec4 envColor = textureCube(cubeMap, R, 0.0);

	// Cheap hdr effect:
    envColor.rgb *= (envColor.r+envColor.g+envColor.b)*hdrMultiplier;

	// Compute refraction ratio (Fresnel):
    float facing = 1.0 - dot(-E, N);
    float refractionRatio = clamp(pow(facing, fresnelPower), 0.0, 1.0);

    // Refracted ray only considers deep and shallow water colors:
    vec4 waterColor = mix(shallowColor, deepColor, facing);

    // Perform linear interpolation between reflection and refraction.
    vec4 color = mix(waterColor, envColor, refractionRatio);

    // generate circular spill
    float low_thresh = 0.2 + 0.1 * sin(atan(pos.y, pos.x) * 5.0 + 25.0 * pos.x);
    float high_thresh = 20.8;
    float c = 1.0 - smoothstep(low_thresh, high_thresh, length(pos));

    float brightness = (0.2126*R.b + 0.7152*R.g + 0.0722*R.b);
    float gradient = smoothstep(0.4, 0.6, brightness);
    vec4 rainbow = frequency_to_rgb(gradient);
    vec3 rainbow3 = mix(vec3(1.0, 1.0, 1.0), rainbow.rgb, 0.5);
    vec3 irColor = mix(color.rgb, rainbow3, 0.75);

    color = vec4(mix(color.rgb, irColor, c), 1.0);


    gl_FragColor = vec4(color.xyz, 0.9);
}
