# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.
import copy
import imgui
import numpy as np
from typing import Optional
from kaolin.render.camera import Camera
from wisp.renderer.gui.imgui.widget_imgui import WidgetImgui, widget
from wisp.renderer.gui.imgui.widget_property_editor import WidgetPropertyEditor
from wisp.renderer.gui.imgui.widget_cameras import WidgetCameraProperties
from wisp.core.colors import black, white, dark_gray
from wisp.framework import WispState, InteractiveRendererState
from wisp.renderer.core.control import FirstPersonCameraMode, TrackballCameraMode, TurntableCameraMode


@widget(InteractiveRendererState)
class WidgetTrainingArguments(WidgetImgui):
    def __init__(self, trainer=None, render_core=None):
        super().__init__()
        self.training_properties = Training(trainer, render_core)
        self.trainer=trainer

        self.available_camera_modes = [FirstPersonCameraMode, TrackballCameraMode, TurntableCameraMode]
        self.available_camera_mode_names = [mode.name() for mode in self.available_camera_modes]

    def paint(self, state: WispState, *args, **kwargs):
        if self.trainer is not None:
            expanded, _ = imgui.collapsing_header("Training arguments", visible=True, flags=imgui.TREE_NODE_DEFAULT_OPEN)
            if expanded:
                self.training_properties.paint(state)
                #imgui.tree_pop()



class Training(WidgetImgui):
    def __init__(self, trainer = None, render_core=None):
        super().__init__()

        self.properties_editor = WidgetPropertyEditor()
        self.trainer = trainer
        self.render_core=render_core

    def paint(self, state: WispState, *args, **kwargs):
        if self.trainer is not None:
            properties = self._training_properties()
            if properties is not None:
                self.properties_editor.paint(state=state, properties=properties)
            #imgui.tree_pop()


    def _training_properties(self):
        def _trans_mult():
            trans_mult = self.trainer.trans_mult

            changed, new_trans_mult = imgui.core.slider_float("##trans_mult",
                                                          value=trans_mult, min_value=0, max_value=1, format='%.3e', power=8)
            if changed and trans_mult != new_trans_mult:
                self.trainer.trans_mult = new_trans_mult

        def _entropy_mult():
            entropy_mult = self.trainer.entropy_mult

            changed, new_entropy_mult = imgui.core.slider_float("##entropy_mult",
                                                          value=entropy_mult, min_value=0, max_value=1, format='%.3e', power=8)
            if changed and entropy_mult != new_entropy_mult:
                self.trainer.entropy_mult = new_entropy_mult
                
        def _emptyUseless_mult():
            emptyUseless_mult = self.trainer.emptyUseless_mult

            changed, new_emptyUseless_mult = imgui.core.slider_float("##emptyUseless_mult",
                                                          value=emptyUseless_mult, min_value=0, max_value=1, format='%.3e', power=8)
            if changed and emptyUseless_mult != new_emptyUseless_mult:
                self.trainer.emptyUseless_mult = new_emptyUseless_mult

        def _emptyUseless_sel():
            emptyUseless_sel = self.trainer.emptyUseless_sel

            changed, new_emptyUseless_sel = imgui.core.slider_float("##emptyUseless_sel",
                                                          value=emptyUseless_sel, min_value=1, max_value=100, power=3)
            if changed and emptyUseless_sel != new_emptyUseless_sel:
                self.trainer.emptyUseless_sel = new_emptyUseless_sel



        def _prune_min_density():
            prune_min_density = 1 - np.exp(-self.trainer.pipeline.nef.prune_min_density)

            changed, new_prune_min_density = imgui.core.slider_float("##prune_min_density",
                                                          value=prune_min_density, min_value=0, max_value=1, format='%.3e', power=8)
            if changed and prune_min_density != new_prune_min_density:
                self.trainer.pipeline.nef.prune_min_density = - np.log(1 - new_prune_min_density + 1e-7)

        def _rendering_radius():
            rendering_radius = np.sqrt(self.trainer.pipeline.nef.render_radius)

            changed, new_rendering_radius = imgui.core.slider_float("##rendering_radius",
                                                          value=rendering_radius, min_value=0, max_value=np.sqrt(3), power=2)
            if changed and rendering_radius != new_rendering_radius:
                self.trainer.pipeline.nef.render_radius = np.square(new_rendering_radius)
                
        def _rendering_threshold():
            rendering_threshold = 1 - np.exp(-list(self.render_core._renderers.values())[0].tracer.rendering_threshold_density)

            changed, new_rendering_threshold = imgui.core.slider_float("##rendering_threshold",
                                                          value=rendering_threshold, min_value=0, max_value=1, power=3)
            if changed and rendering_threshold != new_rendering_threshold:
                list(self.render_core._renderers.values())[0].tracer.rendering_threshold_density = - np.log(1 - new_rendering_threshold + 1e-7)



        properties = {
            'Transient loss mult': _trans_mult,
            'Entropy loss mult': _entropy_mult,
            'Empty loss mult': _emptyUseless_mult,
            'Empty selectivity': _emptyUseless_sel,
            'Prune density':_prune_min_density,
            'Render radius': _rendering_radius,
            'Render treshold': _rendering_threshold,
        }
        return properties

