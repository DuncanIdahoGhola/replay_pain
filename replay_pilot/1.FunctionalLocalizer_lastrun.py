#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on June 25, 2025, at 13:51
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
prefs.hardware['audioLatencyMode'] = '3'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (NOT_STARTED, STARTED, PLAYING, PAUSED,
                                STOPPED, FINISHED, PRESSED, RELEASED, FOREVER, priority)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

import psychopy.iohub as io
from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2024.2.4'
expName = 'FunctionalLocalizer'  # from the Builder filename that created this script
# information about this experiment
expInfo = {
    'participant': 'sub-000',
    'session': '001',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = [1920, 1080]
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s/func_loc/%s_%s_%s' % (expInfo['participant'], expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\labmp-psychopy\\Desktop\\replay_pilot\\1.FunctionalLocalizer_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[0,0,0], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [0,0,0]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    # show a visual indicator if we're in piloting mode
    if PILOTING and prefs.piloting['showPilotingIndicator']:
        win.showPilotingIndicator()
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    
    # Setup iohub keyboard
    ioConfig['Keyboard'] = dict(use_keymap='psychopy')
    
    # Setup iohub experiment
    ioConfig['Experiment'] = dict(filename=thisExp.dataFileName)
    
    # Start ioHub server
    ioServer = io.launchHubServer(window=win, **ioConfig)
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='iohub'
        )
    if deviceManager.getDevice('instr_resp') is None:
        # initialise instr_resp
        instr_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instr_resp',
        )
    if deviceManager.getDevice('key_resp_3') is None:
        # initialise key_resp_3
        key_resp_3 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_3',
        )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    if deviceManager.getDevice('finish_time') is None:
        # initialise finish_time
        finish_time = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='finish_time',
        )
    if deviceManager.getDevice('key_resp_2') is None:
        # initialise key_resp_2
        key_resp_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_2',
        )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], playbackComponents=[]):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    playbackComponents : list, tuple
        List of any components with a `pause` method which need to be paused.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    for comp in playbackComponents:
        comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='ioHub',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    for comp in playbackComponents:
        comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ioHub'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    
    # --- Initialize components for Routine "setup_experiment" ---
    # Run 'Begin Experiment' code from set_up
    import random
    
    #add counterbalance and match/mismatch keys
    participant_id = expInfo['participant']
    last_three_digits = int(participant_id[-3:])
    
    
    if last_three_digits % 2 == 0:
        match_key = '1'
        mismatch_key = '2'
        instr_message = "Vous verrez une image, puis un écran blanc, puis un mot. Appuyez sur '1' si le mot correspond à l’image, et sur '2' s’il ne correspond pas. Appuyez sur la barre d’espace pour commencer."
    else : 
        match_key = '2'
        mismatch_key = '1'
        instr_message = "Vous verrez une image, puis un écran blanc, puis un mot. Appuyez sur '2' si le mot correspond à l’image, et sur '1' s’il ne correspond pas. Appuyez sur la barre d’espace pour commencer."
    
    
    allowed_keys_list = [match_key, mismatch_key]
    
    # Store these for later use in routines
    expInfo['match_key'] = match_key
    expInfo['mismatch_key'] = mismatch_key
    expInfo['allowed_keys_list'] = allowed_keys_list
    
    #open condition file to create list of match per trials
    import pandas as pd
    
    condition_path = f"output_localizers_eff_v2/localizer_conditions_{last_three_digits}.csv"
    
    df_condition = pd.read_csv(condition_path)
    
    if last_three_digits % 2 == 0:
        df_condition['corrAns'] = df_condition['is_match'].map({'match': 1, 'mismatch': 2})
    else:
        df_condition['corrAns'] = df_condition['is_match'].map({'match': 2, 'mismatch': 1})
    
    df_condition.to_csv(condition_path, index=False)
    
    #add trials counter for break 
    
    trials_counter= 0
    instr_text = visual.TextStim(win=win, name='instr_text',
        text=instr_message,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    instr_resp = keyboard.Keyboard(deviceName='instr_resp')
    # Run 'Begin Experiment' code from init_eeg
    
    # EEG
    import struct
    import serial
    import time
    # Trigger box Plus COM port
    adress = 'COM17'
    def init_port(adress):
        port = serial.Serial(adress, baudrate=2000000)
        time.sleep(1) # Wait to make sure it's open
        return port
    
    def write_port(port, pin):
        """
        Turn on one of the ditigal pin
        pin: integer in range 2-13
        """
        # Convert to string
        string = b'' + struct.pack('!B', pin)
        # Write
        port.write(string)
        
    port = init_port(adress)
    port.write([0x00])
    
    # Init EEG
    from psychopy.hardware import brainproducts
    rcs = brainproducts.RemoteControlServer(host='192.168.1.2',timeout=5)
    
    rcs.openRecorder()
    time.sleep(2)
    rcs.workspace = 'C:/Users/labmp-eeg/Desktop/antoine_cognitivemaps/antoine_cognitivemaps.rwksp'
    rcs.participant = expInfo['participant'] + '_' + expInfo['date']
    rcs.expName = 'cmapsfuncloc'
    time.sleep(5)
    rcs.mode = 'monitor' 
    time.sleep(2)
    
    
    
    
    
    # --- Initialize components for Routine "start_eeg" ---
    text_2 = visual.TextStim(win=win, name='text_2',
        text='Vérifier eeg\n',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_3 = keyboard.Keyboard(deviceName='key_resp_3')
    
    # --- Initialize components for Routine "load_images" ---
    preload_current_image_2 = clock.StaticPeriod(win=win, screenHz=expInfo['frameRate'], name='preload_current_image_2')
    text_3 = visual.TextStim(win=win, name='text_3',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.15, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "fixation_2" ---
    fixation = visual.TextStim(win=win, name='fixation',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.15, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "trials_image" ---
    stim_image = visual.ImageStim(
        win=win,
        name='stim_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-2.0)
    
    # --- Initialize components for Routine "trials_word" ---
    stim_word = visual.TextStim(win=win, name='stim_word',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.1, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "blank" ---
    blank_interval = visual.TextStim(win=win, name='blank_interval',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    
    # --- Initialize components for Routine "response" ---
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    text = visual.TextStim(win=win, name='text',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-4.0);
    
    # --- Initialize components for Routine "feedback_routine" ---
    feedback_text_display = visual.TextStim(win=win, name='feedback_text_display',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "iti_routine" ---
    iti_fixation = visual.TextStim(win=win, name='iti_fixation',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "break_2" ---
    timer_text = visual.TextStim(win=win, name='timer_text',
        text='Break in progress, press space to continue',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    finish_time = keyboard.Keyboard(deviceName='finish_time')
    
    # --- Initialize components for Routine "end" ---
    end_text = visual.TextStim(win=win, name='end_text',
        text=' La première partie de l’expérience est terminée.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_2 = keyboard.Keyboard(deviceName='key_resp_2')
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "setup_experiment" ---
    # create an object to store info about Routine setup_experiment
    setup_experiment = data.Routine(
        name='setup_experiment',
        components=[instr_text, instr_resp],
    )
    setup_experiment.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for instr_resp
    instr_resp.keys = []
    instr_resp.rt = []
    _instr_resp_allKeys = []
    # store start times for setup_experiment
    setup_experiment.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    setup_experiment.tStart = globalClock.getTime(format='float')
    setup_experiment.status = STARTED
    thisExp.addData('setup_experiment.started', setup_experiment.tStart)
    setup_experiment.maxDuration = None
    # keep track of which components have finished
    setup_experimentComponents = setup_experiment.components
    for thisComponent in setup_experiment.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "setup_experiment" ---
    setup_experiment.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instr_text* updates
        
        # if instr_text is starting this frame...
        if instr_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr_text.frameNStart = frameN  # exact frame index
            instr_text.tStart = t  # local t and not account for scr refresh
            instr_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr_text.started')
            # update status
            instr_text.status = STARTED
            instr_text.setAutoDraw(True)
        
        # if instr_text is active this frame...
        if instr_text.status == STARTED:
            # update params
            pass
        
        # *instr_resp* updates
        waitOnFlip = False
        
        # if instr_resp is starting this frame...
        if instr_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr_resp.frameNStart = frameN  # exact frame index
            instr_resp.tStart = t  # local t and not account for scr refresh
            instr_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr_resp.started')
            # update status
            instr_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instr_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instr_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instr_resp.status == STARTED and not waitOnFlip:
            theseKeys = instr_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _instr_resp_allKeys.extend(theseKeys)
            if len(_instr_resp_allKeys):
                instr_resp.keys = _instr_resp_allKeys[-1].name  # just the last key pressed
                instr_resp.rt = _instr_resp_allKeys[-1].rt
                instr_resp.duration = _instr_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            setup_experiment.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in setup_experiment.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "setup_experiment" ---
    for thisComponent in setup_experiment.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for setup_experiment
    setup_experiment.tStop = globalClock.getTime(format='float')
    setup_experiment.tStopRefresh = tThisFlipGlobal
    thisExp.addData('setup_experiment.stopped', setup_experiment.tStop)
    # check responses
    if instr_resp.keys in ['', [], None]:  # No response was made
        instr_resp.keys = None
    thisExp.addData('instr_resp.keys',instr_resp.keys)
    if instr_resp.keys != None:  # we had a response
        thisExp.addData('instr_resp.rt', instr_resp.rt)
        thisExp.addData('instr_resp.duration', instr_resp.duration)
    thisExp.nextEntry()
    # the Routine "setup_experiment" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "start_eeg" ---
    # create an object to store info about Routine start_eeg
    start_eeg = data.Routine(
        name='start_eeg',
        components=[text_2, key_resp_3],
    )
    start_eeg.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_3
    key_resp_3.keys = []
    key_resp_3.rt = []
    _key_resp_3_allKeys = []
    # Run 'Begin Routine' code from start_eeg_2
    rec_start = core.monotonicClock.getTime()
    rcs.startRecording()
    # store start times for start_eeg
    start_eeg.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    start_eeg.tStart = globalClock.getTime(format='float')
    start_eeg.status = STARTED
    thisExp.addData('start_eeg.started', start_eeg.tStart)
    start_eeg.maxDuration = None
    # keep track of which components have finished
    start_eegComponents = start_eeg.components
    for thisComponent in start_eeg.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "start_eeg" ---
    start_eeg.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_2* updates
        
        # if text_2 is starting this frame...
        if text_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_2.frameNStart = frameN  # exact frame index
            text_2.tStart = t  # local t and not account for scr refresh
            text_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_2.started')
            # update status
            text_2.status = STARTED
            text_2.setAutoDraw(True)
        
        # if text_2 is active this frame...
        if text_2.status == STARTED:
            # update params
            pass
        
        # if text_2 is stopping this frame...
        if text_2.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_2.tStartRefresh + 1.0-frameTolerance:
                # keep track of stop time/frame for later
                text_2.tStop = t  # not accounting for scr refresh
                text_2.tStopRefresh = tThisFlipGlobal  # on global time
                text_2.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_2.stopped')
                # update status
                text_2.status = FINISHED
                text_2.setAutoDraw(False)
        
        # *key_resp_3* updates
        waitOnFlip = False
        
        # if key_resp_3 is starting this frame...
        if key_resp_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_3.frameNStart = frameN  # exact frame index
            key_resp_3.tStart = t  # local t and not account for scr refresh
            key_resp_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_3.started')
            # update status
            key_resp_3.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_3.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_3.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_3.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_3_allKeys.extend(theseKeys)
            if len(_key_resp_3_allKeys):
                key_resp_3.keys = _key_resp_3_allKeys[-1].name  # just the last key pressed
                key_resp_3.rt = _key_resp_3_allKeys[-1].rt
                key_resp_3.duration = _key_resp_3_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            start_eeg.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in start_eeg.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "start_eeg" ---
    for thisComponent in start_eeg.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for start_eeg
    start_eeg.tStop = globalClock.getTime(format='float')
    start_eeg.tStopRefresh = tThisFlipGlobal
    thisExp.addData('start_eeg.stopped', start_eeg.tStop)
    # check responses
    if key_resp_3.keys in ['', [], None]:  # No response was made
        key_resp_3.keys = None
    thisExp.addData('key_resp_3.keys',key_resp_3.keys)
    if key_resp_3.keys != None:  # we had a response
        thisExp.addData('key_resp_3.rt', key_resp_3.rt)
        thisExp.addData('key_resp_3.duration', key_resp_3.duration)
    thisExp.nextEntry()
    # the Routine "start_eeg" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    inside_loop_trials = data.TrialHandler2(
        name='inside_loop_trials',
        nReps=1.0, 
        method='sequential', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions(condition_path), 
        seed=None, 
    )
    thisExp.addLoop(inside_loop_trials)  # add the loop to the experiment
    thisInside_loop_trial = inside_loop_trials.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisInside_loop_trial.rgb)
    if thisInside_loop_trial != None:
        for paramName in thisInside_loop_trial:
            globals()[paramName] = thisInside_loop_trial[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisInside_loop_trial in inside_loop_trials:
        currentLoop = inside_loop_trials
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisInside_loop_trial.rgb)
        if thisInside_loop_trial != None:
            for paramName in thisInside_loop_trial:
                globals()[paramName] = thisInside_loop_trial[paramName]
        
        # --- Prepare to start Routine "load_images" ---
        # create an object to store info about Routine load_images
        load_images = data.Routine(
            name='load_images',
            components=[preload_current_image_2, text_3],
        )
        load_images.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for load_images
        load_images.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        load_images.tStart = globalClock.getTime(format='float')
        load_images.status = STARTED
        thisExp.addData('load_images.started', load_images.tStart)
        load_images.maxDuration = None
        # keep track of which components have finished
        load_imagesComponents = load_images.components
        for thisComponent in load_images.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "load_images" ---
        # if trial has changed, end Routine now
        if isinstance(inside_loop_trials, data.TrialHandler2) and thisInside_loop_trial.thisN != inside_loop_trials.thisTrial.thisN:
            continueRoutine = False
        load_images.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *text_3* updates
            
            # if text_3 is starting this frame...
            if text_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_3.frameNStart = frameN  # exact frame index
                text_3.tStart = t  # local t and not account for scr refresh
                text_3.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_3, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_3.started')
                # update status
                text_3.status = STARTED
                text_3.setAutoDraw(True)
            
            # if text_3 is active this frame...
            if text_3.status == STARTED:
                # update params
                pass
            
            # if text_3 is stopping this frame...
            if text_3.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text_3.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    text_3.tStop = t  # not accounting for scr refresh
                    text_3.tStopRefresh = tThisFlipGlobal  # on global time
                    text_3.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_3.stopped')
                    # update status
                    text_3.status = FINISHED
                    text_3.setAutoDraw(False)
            # *preload_current_image_2* period
            
            # if preload_current_image_2 is starting this frame...
            if preload_current_image_2.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                preload_current_image_2.frameNStart = frameN  # exact frame index
                preload_current_image_2.tStart = t  # local t and not account for scr refresh
                preload_current_image_2.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(preload_current_image_2, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('preload_current_image_2.started', t)
                # update status
                preload_current_image_2.status = STARTED
                preload_current_image_2.start(0.5)
            elif preload_current_image_2.status == STARTED:  # one frame should pass before updating params and completing
                # Updating other components during *preload_current_image_2*
                stim_image.setImage(image_file)
                stim_word.setText(presented_word)
                blank_interval.setText('')
                # Component updates done
                preload_current_image_2.complete()  # finish the static period
                preload_current_image_2.tStop = preload_current_image_2.tStart + 0.5  # record stop time
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                load_images.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in load_images.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "load_images" ---
        for thisComponent in load_images.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for load_images
        load_images.tStop = globalClock.getTime(format='float')
        load_images.tStopRefresh = tThisFlipGlobal
        thisExp.addData('load_images.stopped', load_images.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if load_images.maxDurationReached:
            routineTimer.addTime(-load_images.maxDuration)
        elif load_images.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        
        # --- Prepare to start Routine "fixation_2" ---
        # create an object to store info about Routine fixation_2
        fixation_2 = data.Routine(
            name='fixation_2',
            components=[fixation],
        )
        fixation_2.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from trial_code
        image_display_dur = 1.0
        word_display_duration = 1.0
        total_response_time_allowed = 2.0
        
        trials_counter = trials_counter + 1
        
        
        # Run 'Begin Routine' code from eeg_fix
        wait_sent = 0
        # store start times for fixation_2
        fixation_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        fixation_2.tStart = globalClock.getTime(format='float')
        fixation_2.status = STARTED
        thisExp.addData('fixation_2.started', fixation_2.tStart)
        fixation_2.maxDuration = None
        # keep track of which components have finished
        fixation_2Components = fixation_2.components
        for thisComponent in fixation_2.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "fixation_2" ---
        # if trial has changed, end Routine now
        if isinstance(inside_loop_trials, data.TrialHandler2) and thisInside_loop_trial.thisN != inside_loop_trials.thisTrial.thisN:
            continueRoutine = False
        fixation_2.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixation* updates
            
            # if fixation is starting this frame...
            if fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixation.frameNStart = frameN  # exact frame index
                fixation.tStart = t  # local t and not account for scr refresh
                fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation.started')
                # update status
                fixation.status = STARTED
                fixation.setAutoDraw(True)
            
            # if fixation is active this frame...
            if fixation.status == STARTED:
                # update params
                pass
            
            # if fixation is stopping this frame...
            if fixation.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation.tStop = t  # not accounting for scr refresh
                    fixation.tStopRefresh = tThisFlipGlobal  # on global time
                    fixation.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation.stopped')
                    # update status
                    fixation.status = FINISHED
                    fixation.setAutoDraw(False)
            # Run 'Each Frame' code from eeg_fix
            if fixation.status == STARTED and wait_sent == 0: #If the stimulus component has started and the trigger has not yet been sent. Change 'stimulus' to match the name of the component you want the trigger to be sent at the same time as
                win.callOnFlip(port.write, data=b'\x01') #Send the trigger, synced to the screen refresh
                #win.callOnFlip(eci_client.send_event, event_type = 'wait') #Send the trigger, synced to the screen refresh
                wait_sent = 1 #The wait has now been sent, so we set this to true to avoid a trigger being sent on each frame
            
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                fixation_2.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in fixation_2.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "fixation_2" ---
        for thisComponent in fixation_2.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for fixation_2
        fixation_2.tStop = globalClock.getTime(format='float')
        fixation_2.tStopRefresh = tThisFlipGlobal
        thisExp.addData('fixation_2.stopped', fixation_2.tStop)
        # Run 'End Routine' code from eeg_fix
        port.write([0x00])
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if fixation_2.maxDurationReached:
            routineTimer.addTime(-fixation_2.maxDuration)
        elif fixation_2.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        
        # --- Prepare to start Routine "trials_image" ---
        # create an object to store info about Routine trials_image
        trials_image = data.Routine(
            name='trials_image',
            components=[stim_image],
        )
        trials_image.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from eeg_image
        wait_sent = 0
        # Run 'Begin Routine' code from photo_sensor
        photodiode_box = visual.Rect(
            win=win,
            width=18,
            height=25,
            fillColor='white',
            lineColor='white',
            pos=(-win.size[0]/2 + 25, -win.size[1]/2 + 25),
            units='pix',
            autoDraw=False
        )
        # store start times for trials_image
        trials_image.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        trials_image.tStart = globalClock.getTime(format='float')
        trials_image.status = STARTED
        thisExp.addData('trials_image.started', trials_image.tStart)
        trials_image.maxDuration = None
        # keep track of which components have finished
        trials_imageComponents = trials_image.components
        for thisComponent in trials_image.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trials_image" ---
        # if trial has changed, end Routine now
        if isinstance(inside_loop_trials, data.TrialHandler2) and thisInside_loop_trial.thisN != inside_loop_trials.thisTrial.thisN:
            continueRoutine = False
        trials_image.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from eeg_image
            if stim_image.status == STARTED and wait_sent == 0: #If the stimulus component has started and the trigger has not yet been sent. Change 'stimulus' to match the name of the component you want the trigger to be sent at the same time as
                win.callOnFlip(port.write, data=b'\x04') #Send the trigger, synced to the screen refresh
                #win.callOnFlip(eci_client.send_event, event_type = 'wait') #Send the trigger, synced to the screen refresh
                wait_sent = 1 #The wait has now been sent, so we set this to true to avoid a trigger being sent on each frame
            
            # Run 'Each Frame' code from photo_sensor
            if stim_image.status == STARTED:
                photodiode_box.autoDraw = True
            else:
                photodiode_box.autoDraw = False
            
            # *stim_image* updates
            
            # if stim_image is starting this frame...
            if stim_image.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                stim_image.frameNStart = frameN  # exact frame index
                stim_image.tStart = t  # local t and not account for scr refresh
                stim_image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(stim_image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'stim_image.started')
                # update status
                stim_image.status = STARTED
                stim_image.setAutoDraw(True)
            
            # if stim_image is active this frame...
            if stim_image.status == STARTED:
                # update params
                pass
            
            # if stim_image is stopping this frame...
            if stim_image.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > stim_image.tStartRefresh + image_display_dur-frameTolerance:
                    # keep track of stop time/frame for later
                    stim_image.tStop = t  # not accounting for scr refresh
                    stim_image.tStopRefresh = tThisFlipGlobal  # on global time
                    stim_image.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'stim_image.stopped')
                    # update status
                    stim_image.status = FINISHED
                    stim_image.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                trials_image.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trials_image.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trials_image" ---
        for thisComponent in trials_image.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for trials_image
        trials_image.tStop = globalClock.getTime(format='float')
        trials_image.tStopRefresh = tThisFlipGlobal
        thisExp.addData('trials_image.stopped', trials_image.tStop)
        # Run 'End Routine' code from eeg_image
        port.write([0x00])
        # Run 'End Routine' code from photo_sensor
        photodiode_box.autoDraw = False
        # the Routine "trials_image" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "trials_word" ---
        # create an object to store info about Routine trials_word
        trials_word = data.Routine(
            name='trials_word',
            components=[stim_word],
        )
        trials_word.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from eeg_word
        wait_sent = 0
        # store start times for trials_word
        trials_word.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        trials_word.tStart = globalClock.getTime(format='float')
        trials_word.status = STARTED
        thisExp.addData('trials_word.started', trials_word.tStart)
        trials_word.maxDuration = None
        # keep track of which components have finished
        trials_wordComponents = trials_word.components
        for thisComponent in trials_word.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "trials_word" ---
        # if trial has changed, end Routine now
        if isinstance(inside_loop_trials, data.TrialHandler2) and thisInside_loop_trial.thisN != inside_loop_trials.thisTrial.thisN:
            continueRoutine = False
        trials_word.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from eeg_word
            if stim_word.status == STARTED and wait_sent == 0: #If the stimulus component has started and the trigger has not yet been sent. Change 'stimulus' to match the name of the component you want the trigger to be sent at the same time as
                win.callOnFlip(port.write, data=b'\x10') #Send the trigger, synced to the screen refresh
                #win.callOnFlip(eci_client.send_event, event_type = 'wait') #Send the trigger, synced to the screen refresh
                wait_sent = 1 #The wait has now been sent, so we set this to true to avoid a trigger being sent on each frame
            
            
            # *stim_word* updates
            
            # if stim_word is starting this frame...
            if stim_word.status == NOT_STARTED and tThisFlip >=  0-frameTolerance:
                # keep track of start time/frame for later
                stim_word.frameNStart = frameN  # exact frame index
                stim_word.tStart = t  # local t and not account for scr refresh
                stim_word.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(stim_word, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'stim_word.started')
                # update status
                stim_word.status = STARTED
                stim_word.setAutoDraw(True)
            
            # if stim_word is active this frame...
            if stim_word.status == STARTED:
                # update params
                pass
            
            # if stim_word is stopping this frame...
            if stim_word.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > stim_word.tStartRefresh + word_display_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    stim_word.tStop = t  # not accounting for scr refresh
                    stim_word.tStopRefresh = tThisFlipGlobal  # on global time
                    stim_word.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'stim_word.stopped')
                    # update status
                    stim_word.status = FINISHED
                    stim_word.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                trials_word.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trials_word.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trials_word" ---
        for thisComponent in trials_word.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for trials_word
        trials_word.tStop = globalClock.getTime(format='float')
        trials_word.tStopRefresh = tThisFlipGlobal
        thisExp.addData('trials_word.stopped', trials_word.tStop)
        # Run 'End Routine' code from eeg_word
        port.write([0x00])
        # the Routine "trials_word" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "blank" ---
        # create an object to store info about Routine blank
        blank = data.Routine(
            name='blank',
            components=[blank_interval],
        )
        blank.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from eeg_blank
        wait_sent = 0
        # Run 'Begin Routine' code from code_2
        import random
        randomizer = random.randint(0, 10)
        
        time_of_blank = 1.0 + (randomizer/10)
        
        # store start times for blank
        blank.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        blank.tStart = globalClock.getTime(format='float')
        blank.status = STARTED
        thisExp.addData('blank.started', blank.tStart)
        blank.maxDuration = None
        # keep track of which components have finished
        blankComponents = blank.components
        for thisComponent in blank.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "blank" ---
        # if trial has changed, end Routine now
        if isinstance(inside_loop_trials, data.TrialHandler2) and thisInside_loop_trial.thisN != inside_loop_trials.thisTrial.thisN:
            continueRoutine = False
        blank.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from eeg_blank
            if blank_interval.status == STARTED and wait_sent == 0: #If the stimulus component has started and the trigger has not yet been sent. Change 'stimulus' to match the name of the component you want the trigger to be sent at the same time as
                win.callOnFlip(port.write, data=b'\x02') #Send the trigger, synced to the screen refresh
                #win.callOnFlip(eci_client.send_event, event_type = 'wait') #Send the trigger, synced to the screen refresh
                wait_sent = 1 #The wait has now been sent, so we set this to true to avoid a trigger being sent on each frame
            
            
            # *blank_interval* updates
            
            # if blank_interval is starting this frame...
            if blank_interval.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                blank_interval.frameNStart = frameN  # exact frame index
                blank_interval.tStart = t  # local t and not account for scr refresh
                blank_interval.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(blank_interval, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'blank_interval.started')
                # update status
                blank_interval.status = STARTED
                blank_interval.setAutoDraw(True)
            
            # if blank_interval is active this frame...
            if blank_interval.status == STARTED:
                # update params
                pass
            
            # if blank_interval is stopping this frame...
            if blank_interval.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > blank_interval.tStartRefresh + time_of_blank-frameTolerance:
                    # keep track of stop time/frame for later
                    blank_interval.tStop = t  # not accounting for scr refresh
                    blank_interval.tStopRefresh = tThisFlipGlobal  # on global time
                    blank_interval.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'blank_interval.stopped')
                    # update status
                    blank_interval.status = FINISHED
                    blank_interval.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                blank.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in blank.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "blank" ---
        for thisComponent in blank.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for blank
        blank.tStop = globalClock.getTime(format='float')
        blank.tStopRefresh = tThisFlipGlobal
        thisExp.addData('blank.stopped', blank.tStop)
        # Run 'End Routine' code from eeg_blank
        port.write([0x00])
        # the Routine "blank" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "response" ---
        # create an object to store info about Routine response
        response = data.Routine(
            name='response',
            components=[key_resp, text],
        )
        response.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from eeg_resp
        wait_sent = 0
        # create starting attributes for key_resp
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        # allowedKeys looks like a variable, so make sure it exists locally
        if 'allowed_keys_list' in globals():
            allowed_keys_list = globals()['allowed_keys_list']
        text.setText(f"Indiquer si le mot correspond à l'image précédente à l'aide de {allowed_keys_list}")
        # store start times for response
        response.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        response.tStart = globalClock.getTime(format='float')
        response.status = STARTED
        thisExp.addData('response.started', response.tStart)
        response.maxDuration = None
        # keep track of which components have finished
        responseComponents = response.components
        for thisComponent in response.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "response" ---
        # if trial has changed, end Routine now
        if isinstance(inside_loop_trials, data.TrialHandler2) and thisInside_loop_trial.thisN != inside_loop_trials.thisTrial.thisN:
            continueRoutine = False
        response.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            # Run 'Each Frame' code from eeg_resp
            if text.status == STARTED and wait_sent == 0: #If the stimulus component has started and the trigger has not yet been sent. Change 'stimulus' to match the name of the component you want the trigger to be sent at the same time as
                win.callOnFlip(port.write, data=b'\x08') #Send the trigger, synced to the screen refresh
                #win.callOnFlip(eci_client.send_event, event_type = 'wait') #Send the trigger, synced to the screen refresh
                wait_sent = 1 #The wait has now been sent, so we set this to true to avoid a trigger being sent on each frame
            
            
            
            # *key_resp* updates
            waitOnFlip = False
            
            # if key_resp is starting this frame...
            if key_resp.status == NOT_STARTED and tThisFlip >= 0-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp.started')
                # update status
                key_resp.status = STARTED
                # allowed keys looks like a variable named `allowed_keys_list`
                if not type(allowed_keys_list) in [list, tuple, np.ndarray]:
                    if not isinstance(allowed_keys_list, str):
                        allowed_keys_list = str(allowed_keys_list)
                    elif not ',' in allowed_keys_list:
                        allowed_keys_list = (allowed_keys_list,)
                    else:
                        allowed_keys_list = eval(allowed_keys_list)
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            
            # if key_resp is stopping this frame...
            if key_resp.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > key_resp.tStartRefresh + total_response_time_allowed-frameTolerance:
                    # keep track of stop time/frame for later
                    key_resp.tStop = t  # not accounting for scr refresh
                    key_resp.tStopRefresh = tThisFlipGlobal  # on global time
                    key_resp.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'key_resp.stopped')
                    # update status
                    key_resp.status = FINISHED
                    key_resp.status = FINISHED
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=list(allowed_keys_list), ignoreKeys=["escape"], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                    key_resp.rt = _key_resp_allKeys[-1].rt
                    key_resp.duration = _key_resp_allKeys[-1].duration
                    # was this correct?
                    if (key_resp.keys == str(corrAns)) or (key_resp.keys == corrAns):
                        key_resp.corr = 1
                    else:
                        key_resp.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            # Run 'Each Frame' code from timer
            t = routineTimer.getTime()
            if t > key_resp.tStart + total_response_time_allowed and not key_resp.keys: 
                key_resp.corr = 0
                thisExp.addData('timeout', 1)
                continueRoutine = False
              
            
            # *text* updates
            
            # if text is starting this frame...
            if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text.frameNStart = frameN  # exact frame index
                text.tStart = t  # local t and not account for scr refresh
                text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text.started')
                # update status
                text.status = STARTED
                text.setAutoDraw(True)
            
            # if text is active this frame...
            if text.status == STARTED:
                # update params
                pass
            
            # if text is stopping this frame...
            if text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > text.tStartRefresh + total_response_time_allowed-frameTolerance:
                    # keep track of stop time/frame for later
                    text.tStop = t  # not accounting for scr refresh
                    text.tStopRefresh = tThisFlipGlobal  # on global time
                    text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text.stopped')
                    # update status
                    text.status = FINISHED
                    text.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                response.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in response.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "response" ---
        for thisComponent in response.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for response
        response.tStop = globalClock.getTime(format='float')
        response.tStopRefresh = tThisFlipGlobal
        thisExp.addData('response.stopped', response.tStop)
        # Run 'End Routine' code from eeg_resp
        port.write([0x00])
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
            # was no response the correct answer?!
            if str(corrAns).lower() == 'none':
               key_resp.corr = 1;  # correct non-response
            else:
               key_resp.corr = 0;  # failed to respond (incorrectly)
        # store data for inside_loop_trials (TrialHandler)
        inside_loop_trials.addData('key_resp.keys',key_resp.keys)
        inside_loop_trials.addData('key_resp.corr', key_resp.corr)
        if key_resp.keys != None:  # we had a response
            inside_loop_trials.addData('key_resp.rt', key_resp.rt)
            inside_loop_trials.addData('key_resp.duration', key_resp.duration)
        # Run 'End Routine' code from save_data
        # Store data
        thisExp.addData('image_presented', image_file)
        thisExp.addData('word_presented', presented_word)
        thisExp.addData('is_match_condition', is_match) # from conditions file
        
        if key_resp.keys:
            thisExp.addData('response_given', key_resp.keys[0])
            thisExp.addData('response_correct', key_resp.corr)
            thisExp.addData('response_rt', key_resp.rt)
        else: # Timeout
            thisExp.addData('response_given', None)
            thisExp.addData('response_correct', 0) # Mark as incorrect or a specific timeout code
            thisExp.addData('response_rt', None)
        
        # Determine feedback message
        feedback_message = ""
        show_feedback_flag = False
        
        if not key_resp.keys: # Timeout
            feedback_message = "Délai de réponse dépassé. Veuillez répondre rapidement."
            show_feedback_flag = True
        elif not key_resp.corr: # Incorrect response
            feedback_message = "Mauvaise réponse" 
            show_feedback_flag = True
        # else: correct response, no feedback_message, show_feedback_flag remains False
        
        # To be used in the feedback routine
        thisExp.addData('feedback_message_to_show', feedback_message)
        thisExp.addData('show_feedback_flag_value', show_feedback_flag)
        # the Routine "response" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "feedback_routine" ---
        # create an object to store info about Routine feedback_routine
        feedback_routine = data.Routine(
            name='feedback_routine',
            components=[feedback_text_display],
        )
        feedback_routine.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        feedback_text_display.setText(feedback_message)
        # Run 'Begin Routine' code from feedback_code
        #skip feedback if right answer
        if not show_feedback_flag:
            continueRoutine = False
        # store start times for feedback_routine
        feedback_routine.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        feedback_routine.tStart = globalClock.getTime(format='float')
        feedback_routine.status = STARTED
        thisExp.addData('feedback_routine.started', feedback_routine.tStart)
        feedback_routine.maxDuration = None
        # keep track of which components have finished
        feedback_routineComponents = feedback_routine.components
        for thisComponent in feedback_routine.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "feedback_routine" ---
        # if trial has changed, end Routine now
        if isinstance(inside_loop_trials, data.TrialHandler2) and thisInside_loop_trial.thisN != inside_loop_trials.thisTrial.thisN:
            continueRoutine = False
        feedback_routine.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.5:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *feedback_text_display* updates
            
            # if feedback_text_display is starting this frame...
            if feedback_text_display.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                feedback_text_display.frameNStart = frameN  # exact frame index
                feedback_text_display.tStart = t  # local t and not account for scr refresh
                feedback_text_display.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(feedback_text_display, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'feedback_text_display.started')
                # update status
                feedback_text_display.status = STARTED
                feedback_text_display.setAutoDraw(True)
            
            # if feedback_text_display is active this frame...
            if feedback_text_display.status == STARTED:
                # update params
                pass
            
            # if feedback_text_display is stopping this frame...
            if feedback_text_display.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > feedback_text_display.tStartRefresh + 1.5-frameTolerance:
                    # keep track of stop time/frame for later
                    feedback_text_display.tStop = t  # not accounting for scr refresh
                    feedback_text_display.tStopRefresh = tThisFlipGlobal  # on global time
                    feedback_text_display.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'feedback_text_display.stopped')
                    # update status
                    feedback_text_display.status = FINISHED
                    feedback_text_display.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                feedback_routine.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in feedback_routine.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "feedback_routine" ---
        for thisComponent in feedback_routine.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for feedback_routine
        feedback_routine.tStop = globalClock.getTime(format='float')
        feedback_routine.tStopRefresh = tThisFlipGlobal
        thisExp.addData('feedback_routine.stopped', feedback_routine.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if feedback_routine.maxDurationReached:
            routineTimer.addTime(-feedback_routine.maxDuration)
        elif feedback_routine.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.500000)
        
        # --- Prepare to start Routine "iti_routine" ---
        # create an object to store info about Routine iti_routine
        iti_routine = data.Routine(
            name='iti_routine',
            components=[iti_fixation],
        )
        iti_routine.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        iti_fixation.setText('')
        # store start times for iti_routine
        iti_routine.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        iti_routine.tStart = globalClock.getTime(format='float')
        iti_routine.status = STARTED
        thisExp.addData('iti_routine.started', iti_routine.tStart)
        iti_routine.maxDuration = None
        # keep track of which components have finished
        iti_routineComponents = iti_routine.components
        for thisComponent in iti_routine.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "iti_routine" ---
        # if trial has changed, end Routine now
        if isinstance(inside_loop_trials, data.TrialHandler2) and thisInside_loop_trial.thisN != inside_loop_trials.thisTrial.thisN:
            continueRoutine = False
        iti_routine.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *iti_fixation* updates
            
            # if iti_fixation is starting this frame...
            if iti_fixation.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                iti_fixation.frameNStart = frameN  # exact frame index
                iti_fixation.tStart = t  # local t and not account for scr refresh
                iti_fixation.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(iti_fixation, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'iti_fixation.started')
                # update status
                iti_fixation.status = STARTED
                iti_fixation.setAutoDraw(True)
            
            # if iti_fixation is active this frame...
            if iti_fixation.status == STARTED:
                # update params
                pass
            
            # if iti_fixation is stopping this frame...
            if iti_fixation.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > iti_fixation.tStartRefresh + iti_duration-frameTolerance:
                    # keep track of stop time/frame for later
                    iti_fixation.tStop = t  # not accounting for scr refresh
                    iti_fixation.tStopRefresh = tThisFlipGlobal  # on global time
                    iti_fixation.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'iti_fixation.stopped')
                    # update status
                    iti_fixation.status = FINISHED
                    iti_fixation.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                iti_routine.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in iti_routine.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "iti_routine" ---
        for thisComponent in iti_routine.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for iti_routine
        iti_routine.tStop = globalClock.getTime(format='float')
        iti_routine.tStopRefresh = tThisFlipGlobal
        thisExp.addData('iti_routine.stopped', iti_routine.tStop)
        # the Routine "iti_routine" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "break_2" ---
        # create an object to store info about Routine break_2
        break_2 = data.Routine(
            name='break_2',
            components=[timer_text, finish_time],
        )
        break_2.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from break_code
        if trials_counter in [74, 128, 222]:
            continueRoutine = True
        else:
            continueRoutine = False
            
        
        # create starting attributes for finish_time
        finish_time.keys = []
        finish_time.rt = []
        _finish_time_allKeys = []
        # store start times for break_2
        break_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        break_2.tStart = globalClock.getTime(format='float')
        break_2.status = STARTED
        thisExp.addData('break_2.started', break_2.tStart)
        break_2.maxDuration = None
        # keep track of which components have finished
        break_2Components = break_2.components
        for thisComponent in break_2.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "break_2" ---
        # if trial has changed, end Routine now
        if isinstance(inside_loop_trials, data.TrialHandler2) and thisInside_loop_trial.thisN != inside_loop_trials.thisTrial.thisN:
            continueRoutine = False
        break_2.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *timer_text* updates
            
            # if timer_text is starting this frame...
            if timer_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                timer_text.frameNStart = frameN  # exact frame index
                timer_text.tStart = t  # local t and not account for scr refresh
                timer_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(timer_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'timer_text.started')
                # update status
                timer_text.status = STARTED
                timer_text.setAutoDraw(True)
            
            # if timer_text is active this frame...
            if timer_text.status == STARTED:
                # update params
                pass
            
            # *finish_time* updates
            waitOnFlip = False
            
            # if finish_time is starting this frame...
            if finish_time.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                finish_time.frameNStart = frameN  # exact frame index
                finish_time.tStart = t  # local t and not account for scr refresh
                finish_time.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(finish_time, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'finish_time.started')
                # update status
                finish_time.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(finish_time.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(finish_time.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if finish_time.status == STARTED and not waitOnFlip:
                theseKeys = finish_time.getKeys(keyList=['p'], ignoreKeys=["escape"], waitRelease=False)
                _finish_time_allKeys.extend(theseKeys)
                if len(_finish_time_allKeys):
                    finish_time.keys = _finish_time_allKeys[-1].name  # just the last key pressed
                    finish_time.rt = _finish_time_allKeys[-1].rt
                    finish_time.duration = _finish_time_allKeys[-1].duration
                    # a response ends the routine
                    continueRoutine = False
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer], 
                    playbackComponents=[]
                )
                # skip the frame we paused on
                continue
            
            # check if all components have finished
            if not continueRoutine:  # a component has requested a forced-end of Routine
                break_2.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in break_2.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "break_2" ---
        for thisComponent in break_2.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for break_2
        break_2.tStop = globalClock.getTime(format='float')
        break_2.tStopRefresh = tThisFlipGlobal
        thisExp.addData('break_2.stopped', break_2.tStop)
        # check responses
        if finish_time.keys in ['', [], None]:  # No response was made
            finish_time.keys = None
        inside_loop_trials.addData('finish_time.keys',finish_time.keys)
        if finish_time.keys != None:  # we had a response
            inside_loop_trials.addData('finish_time.rt', finish_time.rt)
            inside_loop_trials.addData('finish_time.duration', finish_time.duration)
        # the Routine "break_2" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'inside_loop_trials'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    # --- Prepare to start Routine "end" ---
    # create an object to store info about Routine end
    end = data.Routine(
        name='end',
        components=[end_text, key_resp_2],
    )
    end.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_2
    key_resp_2.keys = []
    key_resp_2.rt = []
    _key_resp_2_allKeys = []
    # store start times for end
    end.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    end.tStart = globalClock.getTime(format='float')
    end.status = STARTED
    thisExp.addData('end.started', end.tStart)
    end.maxDuration = None
    # keep track of which components have finished
    endComponents = end.components
    for thisComponent in end.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "end" ---
    end.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *end_text* updates
        
        # if end_text is starting this frame...
        if end_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            end_text.frameNStart = frameN  # exact frame index
            end_text.tStart = t  # local t and not account for scr refresh
            end_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'end_text.started')
            # update status
            end_text.status = STARTED
            end_text.setAutoDraw(True)
        
        # if end_text is active this frame...
        if end_text.status == STARTED:
            # update params
            pass
        
        # if end_text is stopping this frame...
        if end_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > end_text.tStartRefresh + 20-frameTolerance:
                # keep track of stop time/frame for later
                end_text.tStop = t  # not accounting for scr refresh
                end_text.tStopRefresh = tThisFlipGlobal  # on global time
                end_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'end_text.stopped')
                # update status
                end_text.status = FINISHED
                end_text.setAutoDraw(False)
        
        # *key_resp_2* updates
        waitOnFlip = False
        
        # if key_resp_2 is starting this frame...
        if key_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_2.frameNStart = frameN  # exact frame index
            key_resp_2.tStart = t  # local t and not account for scr refresh
            key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_2.started')
            # update status
            key_resp_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_2.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_2_allKeys.extend(theseKeys)
            if len(_key_resp_2_allKeys):
                key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                key_resp_2.duration = _key_resp_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer], 
                playbackComponents=[]
            )
            # skip the frame we paused on
            continue
        
        # check if all components have finished
        if not continueRoutine:  # a component has requested a forced-end of Routine
            end.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in end.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end" ---
    for thisComponent in end.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for end
    end.tStop = globalClock.getTime(format='float')
    end.tStopRefresh = tThisFlipGlobal
    thisExp.addData('end.stopped', end.tStop)
    # check responses
    if key_resp_2.keys in ['', [], None]:  # No response was made
        key_resp_2.keys = None
    thisExp.addData('key_resp_2.keys',key_resp_2.keys)
    if key_resp_2.keys != None:  # we had a response
        thisExp.addData('key_resp_2.rt', key_resp_2.rt)
        thisExp.addData('key_resp_2.duration', key_resp_2.duration)
    thisExp.nextEntry()
    # the Routine "end" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
