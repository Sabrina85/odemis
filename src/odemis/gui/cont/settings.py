#-*- coding: utf-8 -*-
"""
@author: Rinze de Laat

Copyright © 2012-2013 Rinze de Laat, Delmic

This file is part of Odemis.

Odemis is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation, either version 2 of the License, or (at your option) any later
version.

Odemis is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
Odemis. If not, see http://www.gnu.org/licenses/.


### Purpose ###

This module contains classes to control the settings controls in the right
setting column of the user interface.

"""

import collections
import logging
import re

import wx.combo
from wx.lib.pubsub import pub

import odemis.gui
import odemis.gui.comp.text as text
import odemis.gui.img.data as img
import odemis.gui.util.units as utun

from odemis.model import getVAs, NotApplicableError, VigilantAttributeBase
from odemis.gui.comp.foldpanelbar import FoldPanelItem
from odemis.gui.comp.radio import GraphicalRadioButtonControl
from odemis.gui.comp.slider import UnitIntegerSlider, UnitFloatSlider
from odemis.gui.conf.settingspanel import CONFIG
from odemis.gui.model.stream import SpectrumStream
from odemis.gui.util.widgets import VigilantAttributeConnector
from odemis.gui.util.units import readable_str

####### Utility functions #######

def choice_to_str(choice):
    if not isinstance(choice, collections.Iterable):
        choice = [unicode(choice)]
    return u" x ".join([unicode(c) for c in choice])

def traverse(seq_val):
    if isinstance(seq_val, collections.Iterable):
        for value in seq_val:
            for subvalue in traverse(value):
                yield subvalue
    else:
        yield seq_val

def bind_menu(se):
    """
    Add a menu to reset a setting entry to the original (current) value
    :param se: (SettingEntry)

    Note: se must have a valid label, ctrl and va at least
    """
    orig_val = se.va.value

    def reset_value(evt):
        se.va.value = orig_val
        wx.CallAfter(pub.sendMessage, 'setting.changed', setting_ctrl=se.ctrl)

    def show_reset_menu(evt):
        # No menu needed if value hasn't changed
        if se.va.value == orig_val:
            return # TODO: or display it greyed out?

        menu = wx.Menu()
        mi = wx.MenuItem(menu, wx.NewId(), 'Reset value')

        eo = evt.GetEventObject()
        eo.Bind(wx.EVT_MENU, reset_value, mi)

        menu.AppendItem(mi)
        eo.PopupMenu(menu)

    se.ctrl.Bind(wx.EVT_CONTEXT_MENU, show_reset_menu)
    se.label.Bind(wx.EVT_CONTEXT_MENU, show_reset_menu)


####### Classes #######

class SettingEntry(object):
    """
    Represents an setting entry in the panel. It merely associates the VA to
    the widgets that allow to control it.
    """
    # TODO: merge with VAC?
    def __init__(self, name, va=None, comp=None, label=None, ctrl=None, vac=None):
        """
        name (string): name of the va in the component (as-is)
        va (VA): the actual VigilanAttribute
        comp (model.Component): the component that has this VA
        label (wx.LabelTxt): a widget which displays the name of the VA
        ctrl (wx.Window): a widget that allows to change the value
        vac (VigilantAttributeController): the object that ensures the connection
          between the VA and the widget
        """
        self.name = name
        self.va = va
        self.comp = comp
        self.label = label
        self.ctrl = ctrl
        self.vac = vac

    def highlight(self, active=True):
        """
        Highlight the setting entry (ie, the name label becomes bright coloured)
        active (boolean): whether it should be highlighted or not
        """
        if not self.label:
            return

        if active:
            self.label.SetForegroundColour(odemis.gui.FOREGROUND_COLOUR_HIGHLIGHT)
        else:
            self.label.SetForegroundColour(odemis.gui.FOREGROUND_COLOUR)


class SettingsPanel(object):
    """ Settings base class which describes an indirect wrapper for
    FoldPanelItems.

    :param fold_panel: (FoldPanelItem) Parent window
    :param default_msg: (str) Text message which will be shown if the
        SettingPanel does not contain any child windows.
    :param highlight_change: (bool) If set to True, the values will be
        highlighted when they match the cached values.
    NOTE: Do not instantiate this class, but always inherit it.
    """

    def __init__(self, fold_panel, default_msg, highlight_change=False):

        assert isinstance(fold_panel, FoldPanelItem)

        self.fold_panel = fold_panel

        self.panel = wx.Panel(self.fold_panel)


        self.panel.SetBackgroundColour(odemis.gui.BACKGROUND_COLOUR)
        self.panel.SetForegroundColour(odemis.gui.FOREGROUND_COLOUR)

        self.highlight_change = highlight_change

        self._main_sizer = wx.BoxSizer()
        self._gb_sizer = wx.GridBagSizer(0, 0)

        self.panel.SetForegroundColour(odemis.gui.FOREGROUND_COLOUR_DIS)
        self._gb_sizer.Add(wx.StaticText(self.panel, -1, default_msg),
                        (0, 1))
        self.panel.SetForegroundColour(odemis.gui.FOREGROUND_COLOUR)

        self.panel.SetSizer(self._main_sizer)
        self._main_sizer.Add(self._gb_sizer,
                             proportion=1,
                             flag=wx.RIGHT|wx.LEFT|wx.EXPAND,
                             border=5)

        self.fold_panel.add_item(self.panel)

        self._gb_sizer.AddGrowableCol(1)

        self.num_entries = 0
        self.entries = [] # list of SettingEntry

    def pause(self):
        """ Pause VigilantAttributeConnector related control updates """
        for entry in self.entries:
            if entry.vac:
                entry.vac.pause()

    def resume(self):
        """ Pause VigilantAttributeConnector related control updates """
        for entry in self.entries:
            if entry.vac:
                entry.vac.resume()

    def _clear(self):
        # Remove default 'no content' label
        if self.num_entries == 0:
            self.panel.GetChildren()[0].Destroy()

    def _label_to_human(self, label):
        """ Converts a camel-case label into a human readible one
        """
        return re.sub(r"([A-Z])", r" \1", label).capitalize()

    def _determine_default_control(self, value):
        """ Determine the default control to use to represent a vigilant
        attribute in the settings panel.
        """
        if not value:
            logging.warn("No value provided!")
            return odemis.gui.CONTROL_NONE

        if value.readonly:
            return odemis.gui.CONTROL_LABEL
        else:
            try:
                # This statement will raise an exception when no choices are
                # present
                logging.debug("found choices %s", value.choices)

                max_items = 5
                max_len = 5
                # If there are too many choices, or their values are too long
                # in string representation, use a dropdown box

                choices_str = "".join([str(c) for c in value.choices])
                if len(value.choices) < max_items and \
                   len(choices_str) < max_items * max_len:
                    return odemis.gui.CONTROL_RADIO
                else:
                    return odemis.gui.CONTROL_COMBO
            except (AttributeError, NotApplicableError):
                pass

            try:
                # An exception will be raised if no range attribute is found
                logging.debug("found range %s", value.range)
                # TODO: if unit is "s" => scale=exp
                if isinstance(value.value, (int, float)):
                    return odemis.gui.CONTROL_SLIDER
            except (AttributeError, NotApplicableError):
                pass

            # Return default control
            return odemis.gui.CONTROL_TEXT

    def _get_rng_choice_unit(self, va, conf):
        """ Retrieve the range and choices values from the vigilant attribute
        or override them with the values provided in the configuration.
        """

        rng = conf.get("range", (None, None))
        try:
            if rng == (None, None):
                rng = va.range
            else: # merge
                rng = [max(rng[0], va.range[0]), min(rng[1], va.range[1])]
        except (AttributeError, NotApplicableError):
            pass

        choices = conf.get("choices", None)
        try:
            if callable(choices):
                choices = choices(va, conf)
            elif choices is None:
                choices = va.choices
            else: # merge = intersection
                # TODO: if va.range but no va.choices, ensure that
                # choices is within va.range
                choices &= va.choices
        except (AttributeError, NotApplicableError):
            pass

        # Get unit from config, vattribute or use an empty one
        unit =  conf.get('unit', va.unit or "")

        return rng, choices, unit

    def add_label(self, label, value=None):
        """ Adds a label to the settings panel, accompanied by an immutable
        value if one's provided.
        """
        self._clear()
        # Create label
        lbl_ctrl = wx.StaticText(self.panel, -1, "%s" % label)
        self._gb_sizer.Add(lbl_ctrl, (self.num_entries, 0), flag=wx.ALL, border=5)

        value_ctrl = None

        if value:
            self.panel.SetForegroundColour(odemis.gui.FOREGROUND_COLOUR_DIS)

            value_ctrl = wx.StaticText(self.panel, -1, unicode(value))
            self._gb_sizer.Add(value_ctrl, (self.num_entries, 1),
                            flag=wx.ALL, border=5)
            self.panel.SetForegroundColour(odemis.gui.FOREGROUND_COLOUR)

        self.num_entries += 1
        # XXX
        ne = SettingEntry(name=label, label=lbl_ctrl, ctrl=value_ctrl)
        self.entries.append(ne)

    def add_divider(self):
        line = wx.StaticLine(self.panel, size=(-1, 1))
        self._gb_sizer.Add(
                line,
                (self.num_entries, 0),
                span=(1, 2),
                flag=wx.ALL|wx.EXPAND,
                border=5
        )
        self.num_entries += 1


    def add_value(self, name, vigil_attr, comp, conf=None):
        """ Add a name/value pair to the settings panel.

        :param name: (string): name of the value
        :param vigil_attr: (VigilantAttribute)
        :param comp: (Component): the component that contains this VigilantAttribute
        :pram conf: ({}): Configuration items that may override default settings
        """
        assert isinstance(vigil_attr, VigilantAttributeBase)

        # If no conf provided, set it to an empty dictionary
        conf = conf or {}

        # Get the range and choices
        rng, choices, unit = self._get_rng_choice_unit(vigil_attr, conf)

        format = conf.get("format", False)

        if choices:
            if format and all([isinstance(c, (int, float)) for c in choices]):
                choices_formatted, prefix = utun.si_scale_list(choices)
                choices_formatted = [u"%g" % c for c in choices_formatted]
                unit = prefix + unit
            else:
                choices_formatted = [choice_to_str(c) for c in choices]

        # Get the defined type of control or assign a default one
        control_type = conf.get('control_type',
                                self._determine_default_control(vigil_attr))

        # Special case, early stop
        if control_type == odemis.gui.CONTROL_NONE:
            # No value, not even label
            return
        # TODO: if choices has len == 1 => don't provide choice at all
        #  => either display the value as a text, or don't display at all.

        # Remove any 'empty panel' warning
        self._clear()

        # Format label
        label = conf.get('label', self._label_to_human(name))
        # Add the label to the panel
        lbl_ctrl = wx.StaticText(self.panel, -1, "%s" % label)
        self._gb_sizer.Add(lbl_ctrl, (self.num_entries, 0), flag=wx.ALL, border=5)

        # the Vigilant Attribute Connector connects the wx control to the
        # vigilant attribute.
        vac = None

        logging.debug("Adding VA %s", label)
        # Create the needed wxPython controls
        if control_type == odemis.gui.CONTROL_LABEL:
            # Read only value
            # In this case the value need to be transformed into a string

            self.panel.SetForegroundColour(odemis.gui.FOREGROUND_COLOUR_DIS)
            new_ctrl = wx.StaticText(self.panel, -1, size=(200, -1))
            self.panel.SetForegroundColour(odemis.gui.FOREGROUND_COLOUR)

            def format_label(value):
                if isinstance(value, tuple):
                    # Maximum number of chars per value
                    txt = " x ".join(["%s %s" % (v, unit) for v in value])
                else:
                    txt = u"%s %s" % (value, unit)
                new_ctrl.SetLabel(txt)

            vac = VigilantAttributeConnector(vigil_attr,
                                             new_ctrl,
                                             format_label)

        elif control_type == odemis.gui.CONTROL_SLIDER:
            # The slider is accompanied by an extra number text field

            if conf.get('type', "integer") == "integer":
                klass = UnitIntegerSlider
            else:
                klass = UnitFloatSlider

            new_ctrl = klass(self.panel,
                             value=vigil_attr.value,
                             val_range=rng,
                             scale=conf.get('scale', None),
                             unit=unit,
                             t_size=(50, -1))

            vac = VigilantAttributeConnector(vigil_attr,
                                             new_ctrl,
                                             events=wx.EVT_SLIDER)

            new_ctrl.Bind(wx.EVT_SLIDER, self.on_setting_changed)

        elif control_type == odemis.gui.CONTROL_INT:
            if unit == "": # don't display unit prefix if no unit
                unit = None
            new_ctrl = text.UnitIntegerCtrl(self.panel,
                                            style=wx.NO_BORDER,
                                            unit=unit,
                                            min_val=rng[0],
                                            max_val=rng[1],
                                            choices=choices)
            new_ctrl.SetForegroundColour(odemis.gui.FOREGROUND_COLOUR_EDIT)
            new_ctrl.SetBackgroundColour(self.panel.GetBackgroundColour())

            vac = VigilantAttributeConnector(vigil_attr,
                                             new_ctrl,
                                             events=wx.EVT_COMMAND_ENTER)

            new_ctrl.Bind(wx.EVT_TEXT, self.on_setting_changed)
            new_ctrl.Bind(wx.EVT_COMMAND_ENTER, self.on_setting_changed)


        elif control_type == odemis.gui.CONTROL_FLT:
            if unit == "": # don't display unit prefix if no unit
                unit = None
            new_ctrl = text.UnitFloatCtrl(self.panel,
                                          style=wx.NO_BORDER,
                                          unit=unit,
                                          min_val=rng[0],
                                          max_val=rng[1],
                                          choices=choices,
                                          accuracy=6)
            new_ctrl.SetForegroundColour(odemis.gui.FOREGROUND_COLOUR_EDIT)
            new_ctrl.SetBackgroundColour(self.panel.GetBackgroundColour())

            vac = VigilantAttributeConnector(vigil_attr,
                                             new_ctrl,
                                             events=wx.EVT_COMMAND_ENTER)

            new_ctrl.Bind(wx.EVT_COMMAND_ENTER, self.on_setting_changed)
            new_ctrl.Bind(wx.EVT_TEXT, self.on_setting_changed)

        elif control_type == odemis.gui.CONTROL_RADIO:
            new_ctrl = GraphicalRadioButtonControl(self.panel,
                                                   -1,
                                                   size=(-1, 16),
                                                   choices=choices,
                                                   style=wx.NO_BORDER,
                                                   labels=choices_formatted,
                                                   units=unit)

            # TODO: Move to secom specific module/class
            if conf.get('type', None) == "1d_binning":
                # need to convert back and forth between 1D and 2D
                # from 2D to 1D (just pick X)
                def radio_set(value, ctrl=new_ctrl):
                    v = value[0]
                    logging.debug("Setting Radio value to %d", v)
                    # it's fine to set a value not in the choices, it will
                    # just not set any of the buttons.
                    return ctrl.SetValue(v)

                # from 1D to 2D (both identical)
                def radio_get(ctrl=new_ctrl):
                    value = ctrl.GetValue()
                    return (value, value)
            else:
                radio_get = None
                radio_set = None

            vac = VigilantAttributeConnector(vigil_attr,
                                             new_ctrl,
                                             va_2_ctrl=radio_set,
                                             ctrl_2_va=radio_get,
                                             events=wx.EVT_BUTTON)

            new_ctrl.Bind(wx.EVT_BUTTON, self.on_setting_changed)

        elif control_type == odemis.gui.CONTROL_COMBO:

            new_ctrl = wx.combo.OwnerDrawnComboBox(self.panel,
                                                   -1,
                                                   value='',
                                                   pos=(0, 0),
                                                   size=(100, 16),
                                                   style=wx.NO_BORDER |
                                                         wx.CB_DROPDOWN |
                                                         wx.TE_PROCESS_ENTER |
                                                         wx.CB_READONLY)


            # Set colours
            new_ctrl.SetForegroundColour(odemis.gui.FOREGROUND_COLOUR_EDIT)
            new_ctrl.SetBackgroundColour(self.panel.GetBackgroundColour())

            new_ctrl.SetButtonBitmaps(img.getbtn_downBitmap(),
                                      pushButtonBg=False)

            def _eat_event(evt):
                """ Quick and dirty empty function used to 'eat'
                mouse wheel events
                """
                # TODO: This solution only makes sure that the control's value
                # doesn't accidentally get altered when it gets hit by a mouse
                # wheel event. However, it also stop the event from propagating
                # so the containing scrolled window will not scroll either.
                # (If the event is skipped, the control will change value again)
                pass

            new_ctrl.Bind(wx.EVT_MOUSEWHEEL, _eat_event)

            # Set choices
            for choice, formatted in zip(choices, choices_formatted):
                new_ctrl.Append(u"%s %s" % (formatted, unit), choice)

            # A small wrapper function makes sure that the value can
            # be set by passing the actual value (As opposed to the text label)
            def cb_set(value, ctrl=new_ctrl):
                for i in range(ctrl.Count):
                    if ctrl.GetClientData(i) == value:
                        logging.debug("Setting ComboBox value to %s", ctrl.Items[i])
                        return ctrl.SetValue(ctrl.Items[i])
                logging.warning("No matching label found for value %s!", value)

            # equivalent wrapper function to retrieve the actual value
            def cb_get(ctrl=new_ctrl):
                value = ctrl.GetValue()
                for i in range(ctrl.Count):
                    if ctrl.Items[i] == value:
                        logging.debug("Getting ComboBox value %s",
                                  ctrl.GetClientData(i))
                        return ctrl.GetClientData(i)


            vac = VigilantAttributeConnector(
                    vigil_attr,
                    new_ctrl,
                    va_2_ctrl=cb_set,
                    ctrl_2_va=cb_get,
                    events=(wx.EVT_COMBOBOX, wx.EVT_TEXT_ENTER))

            new_ctrl.Bind(wx.EVT_COMBOBOX, self.on_setting_changed)
            new_ctrl.Bind(wx.EVT_TEXT_ENTER, self.on_setting_changed)


        else:
            txt = readable_str(vigil_attr.value, unit)
            new_ctrl = wx.StaticText(self.panel, -1, txt)

        #if self.highlight_change and hasattr(new_ctrl, 'SetValue'):
        #    new_ctrl.SetForegroundColour(FOREGROUND_COLOUR_HIGHLIGHT)

        self._gb_sizer.Add(new_ctrl, (self.num_entries, 1),
                        flag=wx.ALL|wx.EXPAND, border=5)

        ne = SettingEntry(name, vigil_attr, comp, lbl_ctrl, new_ctrl, vac)
        self.entries.append(ne)
        self.num_entries += 1

        if self.highlight_change:
            bind_menu(ne)

        self.fold_panel.Parent.Layout()

    def on_setting_changed(self, evt):
        logging.debug("Setting has changed")
        evt_obj = evt.GetEventObject()
        # Make sure the message is sent form the main thread
        wx.CallAfter(pub.sendMessage, 'setting.changed', setting_ctrl=evt_obj)
        evt.Skip()

class SemSettingsPanel(SettingsPanel):
    pass

class OpticalSettingsPanel(SettingsPanel):
    pass

class AngularSettingsPanel(SettingsPanel):
    pass

class SpectrumSettingsPanel(SettingsPanel):
    pass

class SettingsBarController(object):
    """ The main controller class for the settings panel in the live view and
    acquisition frame.

    This class can be used to set, get and otherwise manipulate the content
    of the setting panel.
    """

    def __init__(self, interface_model, parent_frame, highlight_change=False):
        self._interface_model = interface_model
        self.settings_panels = []


    def pause(self):
        """ Pause VigilantAttributeConnector related control updates """
        for panel in self.settings_panels:
            panel.pause()

    def resume(self):
        """ Resume VigilantAttributeConnector related control updates """
        for panel in self.settings_panels:
            panel.resume()

    @property
    def entries(self):
        """
        A list of all the setting entries of all the panels
        """
        entries = []
        for panel in self.settings_panels:
            entries.extend(panel.entries)
        return entries

    def add_component(self, label, comp, panel):

        self.settings_panels.append(panel)

        try:
            panel.add_label(label, comp.name)
            vigil_attrs = getVAs(comp)
            for name, value in vigil_attrs.items():
                if comp.role in CONFIG and name in CONFIG[comp.role]:
                    conf = CONFIG[comp.role][name]
                else:
                    logging.warn("No config found for %s: %s", comp.role, name)
                    conf = None
                panel.add_value(name, value, comp, conf)
        except TypeError:
            msg = "Error adding %s setting for: %s"
            logging.exception(msg, comp.name, name)

    def add_stream(self, stream):
        pass


class SecomSettingsController(SettingsBarController):

    def __init__(self, parent_frame, microscope_model, highlight_change=False):
        super(SecomSettingsController, self).__init__(microscope_model,
                                                      highlight_change)

        self._sem_panel = SemSettingsPanel(
                                    parent_frame.fp_settings_secom_sem,
                                    "No SEM found",
                                    highlight_change)

        self._optical_panel = OpticalSettingsPanel(
                                    parent_frame.fp_settings_secom_optical,
                                    "No optical microscope found",
                                    highlight_change)

        # Query Odemis daemon (Should move this to separate thread)
        if microscope_model.ccd:
            self.add_component("Camera", microscope_model.ccd, self._optical_panel)
        # TODO allow to change light.power

        if microscope_model.ebeam:
            self.add_component("SEM", microscope_model.ebeam, self._sem_panel )


class SparcSettingsController(SettingsBarController):

    def __init__(self, parent_frame, microscope_model, streams, highlight_change=False):
        super(SparcSettingsController, self).__init__(microscope_model,
                                                      highlight_change)

        self._sem_panel = SemSettingsPanel(
                                    parent_frame.fp_settings_sparc_sem,
                                    "No SEM found",
                                    highlight_change)
        self._angular_panel = AngularSettingsPanel(
                                    parent_frame.fp_settings_sparc_angular,
                                    "No angular camera found",
                                    highlight_change)
        self._spectrum_panel = SpectrumSettingsPanel(
                                    parent_frame.fp_settings_sparc_spectrum,
                                    "No spectrometer found",
                                    highlight_change)
        
        if microscope_model.ebeam:
            self.add_component(
                    "SEM",
                    microscope_model.ebeam,
                    self._sem_panel
            )

        if microscope_model.spectrometer:
            self.add_component(
                    "Spectrometer",
                    microscope_model.spectrometer,
                    self._spectrum_panel
            )

            spectrum_streams = [s for s in streams if isinstance(s, SpectrumStream)]

            if spectrum_streams:
                self._spectrum_panel.add_divider()

                for spectrum_stream in spectrum_streams:
                    self._spectrum_panel.add_value(
                            "repetition",
                            spectrum_stream.repetition,
                            None,  #component
                            CONFIG["spectrometer"]["repetition"])

                    # Added for debug only
                    self._spectrum_panel.add_value(
                            "roi (debug)",
                            spectrum_stream.roi,
                            None,  #component
                            CONFIG["spectrometer"]["roi"])
        else:
            parent_frame.fp_settings_sparc_spectrum.Hide()

        if microscope_model.ccd:
            self.add_component(
                    "Angular Camera",
                    microscope_model.ccd,
                    self._angular_panel
            )
        else:
            parent_frame.fp_settings_sparc_angular.Hide()
