#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on juin 16, 2025, at 10:15
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
expName = 'learn_probe'  # from the Builder filename that created this script
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
    filename = u'data/%s/learn_prob/%s_%s_%s' % (expInfo['participant'], expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\cbant\\OneDrive\\Bureau\\git\\replay_pain\\replay_pilot\\2.learn_probe_lastrun.py',
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
    if deviceManager.getDevice('instr_end') is None:
        # initialise instr_end
        instr_end = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='instr_end',
        )
    if deviceManager.getDevice('exit_eeg_text') is None:
        # initialise exit_eeg_text
        exit_eeg_text = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='exit_eeg_text',
        )
    if deviceManager.getDevice('key_resp') is None:
        # initialise key_resp
        key_resp = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp',
        )
    if deviceManager.getDevice('end_probe_instr') is None:
        # initialise end_probe_instr
        end_probe_instr = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='end_probe_instr',
        )
    if deviceManager.getDevice('probe_resp_test') is None:
        # initialise probe_resp_test
        probe_resp_test = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='probe_resp_test',
        )
    if deviceManager.getDevice('key_resp_2') is None:
        # initialise key_resp_2
        key_resp_2 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_2',
        )
    if deviceManager.getDevice('end_exp_1') is None:
        # initialise end_exp_1
        end_exp_1 = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='end_exp_1',
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
    
    # --- Initialize components for Routine "instructions" ---
    instructions_main = visual.TextStim(win=win, name='instructions_main',
        text='Add instructions here\n\nspace to end',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    instr_end = keyboard.Keyboard(deviceName='instr_end')
    # Run 'Begin Experiment' code from exp_beginning
    import random
    
    #add counterbalance and allowed keys
    participant_id = expInfo['participant']
    last_three_digits = int(participant_id[-3:])
    
    if last_three_digits % 2 == 0:
        sequence_good = '1'
        sequence_bad = '2'
        instr_message = " Vous verrez une image. Vous devrez ensuite vous imaginer la séquence complète des images qui la suivent. Une seconde image apparaîtra ensuite. Appuyez sur '1' si cette image est la prochaine dans la séquence, et sur '2' si ce n’est pas le cas. Appuyez sur la barre d’espace pour commencer."
    else : 
        sequence_good = '2'
        sequence_bad = '1'
        instr_message = " Vous verrez une image. Vous devrez ensuite vous imaginer la séquence complète des images qui la suivent. Une seconde image apparaîtra ensuite. Appuyez sur '2' si cette image est la prochaine dans la séquence, et sur '1' si ce n’est pas le cas. Appuyez sur la barre d’espace pour commencer."
    
    allowed_keys_list = [sequence_good, sequence_bad]
    
    # Store these for later use in routines
    expInfo['sequence_good_key'] = sequence_good
    expInfo['sequence_bad_key'] = sequence_bad
    expInfo['allowed_keys_list'] = allowed_keys_list
    
    #open condition file to create list of sequence_good per trials
    import pandas as pd
    
    condition_probe = f"conditions_probe/probe_learning_conditions_{last_three_digits}.csv"
    
    condition_learn = f"conditions_learning/learning_conditions_{last_three_digits}.csv"
    
    answer_counter = 0
    
    # --- Initialize components for Routine "start_eeg" ---
    eeg_start_text = visual.TextStim(win=win, name='eeg_start_text',
        text='Vérifier EEG',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    exit_eeg_text = keyboard.Keyboard(deviceName='exit_eeg_text')
    
    # --- Initialize components for Routine "instructions_learning" ---
    text = visual.TextStim(win=win, name='text',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard(deviceName='key_resp')
    
    # --- Initialize components for Routine "image_load_2" ---
    Image_load_text = visual.TextStim(win=win, name='Image_load_text',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.15, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    load_images_2 = clock.StaticPeriod(win=win, screenHz=expInfo['frameRate'], name='load_images_2')
    
    # --- Initialize components for Routine "fix_learning" ---
    fix_learn_cross = visual.TextStim(win=win, name='fix_learn_cross',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.15, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "image_learn_1" ---
    first_of_pair = visual.ImageStim(
        win=win,
        name='first_of_pair', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "break_img_learn" ---
    break_random = visual.TextStim(win=win, name='break_random',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "image_learn_2" ---
    second_of_pair = visual.ImageStim(
        win=win,
        name='second_of_pair', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "routine_5s_break" ---
    five_s_break = visual.TextStim(win=win, name='five_s_break',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "probe_instr" ---
    probe_instr_text = visual.TextStim(win=win, name='probe_instr_text',
        text=instr_message,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    end_probe_instr = keyboard.Keyboard(deviceName='end_probe_instr')
    
    # --- Initialize components for Routine "image_loader" ---
    empty_text = visual.TextStim(win=win, name='empty_text',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.15, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    image_load = clock.StaticPeriod(win=win, screenHz=expInfo['frameRate'], name='image_load')
    
    # --- Initialize components for Routine "probe_cue" ---
    probe_cue_1 = visual.ImageStim(
        win=win,
        name='probe_cue_1', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "probe_imagination" ---
    imagination_woot = visual.TextStim(win=win, name='imagination_woot',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "probe_test" ---
    prob_test_img = visual.ImageStim(
        win=win,
        name='prob_test_img', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "probe_resp" ---
    probe_resp_test = keyboard.Keyboard(deviceName='probe_resp_test')
    text_4 = visual.TextStim(win=win, name='text_4',
        text='answer',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    
    # --- Initialize components for Routine "probe_break" ---
    text_2 = visual.TextStim(win=win, name='text_2',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    # Run 'Begin Experiment' code from save_data
    
    
    
    # --- Initialize components for Routine "end_of_block" ---
    text_3 = visual.TextStim(win=win, name='text_3',
        text='End of block text\n\nSpace to quit',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_2 = keyboard.Keyboard(deviceName='key_resp_2')
    
    # --- Initialize components for Routine "end_exp" ---
    text_end = visual.TextStim(win=win, name='text_end',
        text='Thank you for participating! ',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    end_exp_1 = keyboard.Keyboard(deviceName='end_exp_1')
    # Run 'Begin Experiment' code from last_data_save
    #count participants answers percentages
    
    percentage_right_answer = (answer_counter/36) * 100
    thisExp.addData('%_answer', percentage_right_answer)
    
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
    
    # --- Prepare to start Routine "instructions" ---
    # create an object to store info about Routine instructions
    instructions = data.Routine(
        name='instructions',
        components=[instructions_main, instr_end],
    )
    instructions.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for instr_end
    instr_end.keys = []
    instr_end.rt = []
    _instr_end_allKeys = []
    # store start times for instructions
    instructions.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instructions.tStart = globalClock.getTime(format='float')
    instructions.status = STARTED
    thisExp.addData('instructions.started', instructions.tStart)
    instructions.maxDuration = None
    # keep track of which components have finished
    instructionsComponents = instructions.components
    for thisComponent in instructions.components:
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
    
    # --- Run Routine "instructions" ---
    instructions.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *instructions_main* updates
        
        # if instructions_main is starting this frame...
        if instructions_main.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instructions_main.frameNStart = frameN  # exact frame index
            instructions_main.tStart = t  # local t and not account for scr refresh
            instructions_main.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instructions_main, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instructions_main.started')
            # update status
            instructions_main.status = STARTED
            instructions_main.setAutoDraw(True)
        
        # if instructions_main is active this frame...
        if instructions_main.status == STARTED:
            # update params
            pass
        
        # *instr_end* updates
        waitOnFlip = False
        
        # if instr_end is starting this frame...
        if instr_end.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            instr_end.frameNStart = frameN  # exact frame index
            instr_end.tStart = t  # local t and not account for scr refresh
            instr_end.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(instr_end, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'instr_end.started')
            # update status
            instr_end.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(instr_end.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(instr_end.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if instr_end.status == STARTED and not waitOnFlip:
            theseKeys = instr_end.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _instr_end_allKeys.extend(theseKeys)
            if len(_instr_end_allKeys):
                instr_end.keys = _instr_end_allKeys[-1].name  # just the last key pressed
                instr_end.rt = _instr_end_allKeys[-1].rt
                instr_end.duration = _instr_end_allKeys[-1].duration
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
            instructions.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instructions.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instructions" ---
    for thisComponent in instructions.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instructions
    instructions.tStop = globalClock.getTime(format='float')
    instructions.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instructions.stopped', instructions.tStop)
    # check responses
    if instr_end.keys in ['', [], None]:  # No response was made
        instr_end.keys = None
    thisExp.addData('instr_end.keys',instr_end.keys)
    if instr_end.keys != None:  # we had a response
        thisExp.addData('instr_end.rt', instr_end.rt)
        thisExp.addData('instr_end.duration', instr_end.duration)
    thisExp.nextEntry()
    # the Routine "instructions" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "start_eeg" ---
    # create an object to store info about Routine start_eeg
    start_eeg = data.Routine(
        name='start_eeg',
        components=[eeg_start_text, exit_eeg_text],
    )
    start_eeg.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for exit_eeg_text
    exit_eeg_text.keys = []
    exit_eeg_text.rt = []
    _exit_eeg_text_allKeys = []
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
        
        # *eeg_start_text* updates
        
        # if eeg_start_text is starting this frame...
        if eeg_start_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            eeg_start_text.frameNStart = frameN  # exact frame index
            eeg_start_text.tStart = t  # local t and not account for scr refresh
            eeg_start_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(eeg_start_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'eeg_start_text.started')
            # update status
            eeg_start_text.status = STARTED
            eeg_start_text.setAutoDraw(True)
        
        # if eeg_start_text is active this frame...
        if eeg_start_text.status == STARTED:
            # update params
            pass
        
        # if eeg_start_text is stopping this frame...
        if eeg_start_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > eeg_start_text.tStartRefresh + 5-frameTolerance:
                # keep track of stop time/frame for later
                eeg_start_text.tStop = t  # not accounting for scr refresh
                eeg_start_text.tStopRefresh = tThisFlipGlobal  # on global time
                eeg_start_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'eeg_start_text.stopped')
                # update status
                eeg_start_text.status = FINISHED
                eeg_start_text.setAutoDraw(False)
        
        # *exit_eeg_text* updates
        waitOnFlip = False
        
        # if exit_eeg_text is starting this frame...
        if exit_eeg_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            exit_eeg_text.frameNStart = frameN  # exact frame index
            exit_eeg_text.tStart = t  # local t and not account for scr refresh
            exit_eeg_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(exit_eeg_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'exit_eeg_text.started')
            # update status
            exit_eeg_text.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(exit_eeg_text.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(exit_eeg_text.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if exit_eeg_text.status == STARTED and not waitOnFlip:
            theseKeys = exit_eeg_text.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _exit_eeg_text_allKeys.extend(theseKeys)
            if len(_exit_eeg_text_allKeys):
                exit_eeg_text.keys = _exit_eeg_text_allKeys[-1].name  # just the last key pressed
                exit_eeg_text.rt = _exit_eeg_text_allKeys[-1].rt
                exit_eeg_text.duration = _exit_eeg_text_allKeys[-1].duration
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
    if exit_eeg_text.keys in ['', [], None]:  # No response was made
        exit_eeg_text.keys = None
    thisExp.addData('exit_eeg_text.keys',exit_eeg_text.keys)
    if exit_eeg_text.keys != None:  # we had a response
        thisExp.addData('exit_eeg_text.rt', exit_eeg_text.rt)
        thisExp.addData('exit_eeg_text.duration', exit_eeg_text.duration)
    thisExp.nextEntry()
    # the Routine "start_eeg" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    blocks_loop = data.TrialHandler2(
        name='blocks_loop',
        nReps=3.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=[None], 
        seed=None, 
    )
    thisExp.addLoop(blocks_loop)  # add the loop to the experiment
    thisBlocks_loop = blocks_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisBlocks_loop.rgb)
    if thisBlocks_loop != None:
        for paramName in thisBlocks_loop:
            globals()[paramName] = thisBlocks_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisBlocks_loop in blocks_loop:
        currentLoop = blocks_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisBlocks_loop.rgb)
        if thisBlocks_loop != None:
            for paramName in thisBlocks_loop:
                globals()[paramName] = thisBlocks_loop[paramName]
        
        # --- Prepare to start Routine "instructions_learning" ---
        # create an object to store info about Routine instructions_learning
        instructions_learning = data.Routine(
            name='instructions_learning',
            components=[text, key_resp],
        )
        instructions_learning.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        text.setText('add learning instr here')
        # create starting attributes for key_resp
        key_resp.keys = []
        key_resp.rt = []
        _key_resp_allKeys = []
        # store start times for instructions_learning
        instructions_learning.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        instructions_learning.tStart = globalClock.getTime(format='float')
        instructions_learning.status = STARTED
        thisExp.addData('instructions_learning.started', instructions_learning.tStart)
        instructions_learning.maxDuration = None
        # keep track of which components have finished
        instructions_learningComponents = instructions_learning.components
        for thisComponent in instructions_learning.components:
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
        
        # --- Run Routine "instructions_learning" ---
        # if trial has changed, end Routine now
        if isinstance(blocks_loop, data.TrialHandler2) and thisBlocks_loop.thisN != blocks_loop.thisTrial.thisN:
            continueRoutine = False
        instructions_learning.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
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
            
            # *key_resp* updates
            waitOnFlip = False
            
            # if key_resp is starting this frame...
            if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp.frameNStart = frameN  # exact frame index
                key_resp.tStart = t  # local t and not account for scr refresh
                key_resp.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp.started')
                # update status
                key_resp.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp.status == STARTED and not waitOnFlip:
                theseKeys = key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_allKeys.extend(theseKeys)
                if len(_key_resp_allKeys):
                    key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                    key_resp.rt = _key_resp_allKeys[-1].rt
                    key_resp.duration = _key_resp_allKeys[-1].duration
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
                instructions_learning.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in instructions_learning.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "instructions_learning" ---
        for thisComponent in instructions_learning.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for instructions_learning
        instructions_learning.tStop = globalClock.getTime(format='float')
        instructions_learning.tStopRefresh = tThisFlipGlobal
        thisExp.addData('instructions_learning.stopped', instructions_learning.tStop)
        # check responses
        if key_resp.keys in ['', [], None]:  # No response was made
            key_resp.keys = None
        blocks_loop.addData('key_resp.keys',key_resp.keys)
        if key_resp.keys != None:  # we had a response
            blocks_loop.addData('key_resp.rt', key_resp.rt)
            blocks_loop.addData('key_resp.duration', key_resp.duration)
        # the Routine "instructions_learning" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        learn_images = data.TrialHandler2(
            name='learn_images',
            nReps=3.0, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions(condition_learn), 
            seed=None, 
        )
        thisExp.addLoop(learn_images)  # add the loop to the experiment
        thisLearn_image = learn_images.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisLearn_image.rgb)
        if thisLearn_image != None:
            for paramName in thisLearn_image:
                globals()[paramName] = thisLearn_image[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisLearn_image in learn_images:
            currentLoop = learn_images
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisLearn_image.rgb)
            if thisLearn_image != None:
                for paramName in thisLearn_image:
                    globals()[paramName] = thisLearn_image[paramName]
            
            # --- Prepare to start Routine "image_load_2" ---
            # create an object to store info about Routine image_load_2
            image_load_2 = data.Routine(
                name='image_load_2',
                components=[Image_load_text, load_images_2],
            )
            image_load_2.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for image_load_2
            image_load_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            image_load_2.tStart = globalClock.getTime(format='float')
            image_load_2.status = STARTED
            thisExp.addData('image_load_2.started', image_load_2.tStart)
            image_load_2.maxDuration = None
            # keep track of which components have finished
            image_load_2Components = image_load_2.components
            for thisComponent in image_load_2.components:
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
            
            # --- Run Routine "image_load_2" ---
            # if trial has changed, end Routine now
            if isinstance(learn_images, data.TrialHandler2) and thisLearn_image.thisN != learn_images.thisTrial.thisN:
                continueRoutine = False
            image_load_2.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.3:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *Image_load_text* updates
                
                # if Image_load_text is starting this frame...
                if Image_load_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    Image_load_text.frameNStart = frameN  # exact frame index
                    Image_load_text.tStart = t  # local t and not account for scr refresh
                    Image_load_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(Image_load_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'Image_load_text.started')
                    # update status
                    Image_load_text.status = STARTED
                    Image_load_text.setAutoDraw(True)
                
                # if Image_load_text is active this frame...
                if Image_load_text.status == STARTED:
                    # update params
                    pass
                
                # if Image_load_text is stopping this frame...
                if Image_load_text.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > Image_load_text.tStartRefresh + 0.3-frameTolerance:
                        # keep track of stop time/frame for later
                        Image_load_text.tStop = t  # not accounting for scr refresh
                        Image_load_text.tStopRefresh = tThisFlipGlobal  # on global time
                        Image_load_text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'Image_load_text.stopped')
                        # update status
                        Image_load_text.status = FINISHED
                        Image_load_text.setAutoDraw(False)
                # *load_images_2* period
                
                # if load_images_2 is starting this frame...
                if load_images_2.status == NOT_STARTED and t >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    load_images_2.frameNStart = frameN  # exact frame index
                    load_images_2.tStart = t  # local t and not account for scr refresh
                    load_images_2.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(load_images_2, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.addData('load_images_2.started', t)
                    # update status
                    load_images_2.status = STARTED
                    load_images_2.start(0.3)
                elif load_images_2.status == STARTED:  # one frame should pass before updating params and completing
                    # Updating other components during *load_images_2*
                    first_of_pair.setImage(stim1_img)
                    break_random.setText('')
                    second_of_pair.setImage(stim2_img)
                    # Component updates done
                    load_images_2.complete()  # finish the static period
                    load_images_2.tStop = load_images_2.tStart + 0.3  # record stop time
                
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
                    image_load_2.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in image_load_2.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "image_load_2" ---
            for thisComponent in image_load_2.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for image_load_2
            image_load_2.tStop = globalClock.getTime(format='float')
            image_load_2.tStopRefresh = tThisFlipGlobal
            thisExp.addData('image_load_2.stopped', image_load_2.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if image_load_2.maxDurationReached:
                routineTimer.addTime(-image_load_2.maxDuration)
            elif image_load_2.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.300000)
            
            # --- Prepare to start Routine "fix_learning" ---
            # create an object to store info about Routine fix_learning
            fix_learning = data.Routine(
                name='fix_learning',
                components=[fix_learn_cross],
            )
            fix_learning.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for fix_learning
            fix_learning.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            fix_learning.tStart = globalClock.getTime(format='float')
            fix_learning.status = STARTED
            thisExp.addData('fix_learning.started', fix_learning.tStart)
            fix_learning.maxDuration = None
            # keep track of which components have finished
            fix_learningComponents = fix_learning.components
            for thisComponent in fix_learning.components:
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
            
            # --- Run Routine "fix_learning" ---
            # if trial has changed, end Routine now
            if isinstance(learn_images, data.TrialHandler2) and thisLearn_image.thisN != learn_images.thisTrial.thisN:
                continueRoutine = False
            fix_learning.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.3:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *fix_learn_cross* updates
                
                # if fix_learn_cross is starting this frame...
                if fix_learn_cross.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    fix_learn_cross.frameNStart = frameN  # exact frame index
                    fix_learn_cross.tStart = t  # local t and not account for scr refresh
                    fix_learn_cross.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(fix_learn_cross, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fix_learn_cross.started')
                    # update status
                    fix_learn_cross.status = STARTED
                    fix_learn_cross.setAutoDraw(True)
                
                # if fix_learn_cross is active this frame...
                if fix_learn_cross.status == STARTED:
                    # update params
                    pass
                
                # if fix_learn_cross is stopping this frame...
                if fix_learn_cross.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > fix_learn_cross.tStartRefresh + 0.3-frameTolerance:
                        # keep track of stop time/frame for later
                        fix_learn_cross.tStop = t  # not accounting for scr refresh
                        fix_learn_cross.tStopRefresh = tThisFlipGlobal  # on global time
                        fix_learn_cross.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'fix_learn_cross.stopped')
                        # update status
                        fix_learn_cross.status = FINISHED
                        fix_learn_cross.setAutoDraw(False)
                
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
                    fix_learning.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in fix_learning.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "fix_learning" ---
            for thisComponent in fix_learning.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for fix_learning
            fix_learning.tStop = globalClock.getTime(format='float')
            fix_learning.tStopRefresh = tThisFlipGlobal
            thisExp.addData('fix_learning.stopped', fix_learning.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if fix_learning.maxDurationReached:
                routineTimer.addTime(-fix_learning.maxDuration)
            elif fix_learning.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.300000)
            
            # --- Prepare to start Routine "image_learn_1" ---
            # create an object to store info about Routine image_learn_1
            image_learn_1 = data.Routine(
                name='image_learn_1',
                components=[first_of_pair],
            )
            image_learn_1.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for image_learn_1
            image_learn_1.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            image_learn_1.tStart = globalClock.getTime(format='float')
            image_learn_1.status = STARTED
            thisExp.addData('image_learn_1.started', image_learn_1.tStart)
            image_learn_1.maxDuration = None
            # keep track of which components have finished
            image_learn_1Components = image_learn_1.components
            for thisComponent in image_learn_1.components:
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
            
            # --- Run Routine "image_learn_1" ---
            # if trial has changed, end Routine now
            if isinstance(learn_images, data.TrialHandler2) and thisLearn_image.thisN != learn_images.thisTrial.thisN:
                continueRoutine = False
            image_learn_1.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.5:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *first_of_pair* updates
                
                # if first_of_pair is starting this frame...
                if first_of_pair.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    first_of_pair.frameNStart = frameN  # exact frame index
                    first_of_pair.tStart = t  # local t and not account for scr refresh
                    first_of_pair.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(first_of_pair, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'first_of_pair.started')
                    # update status
                    first_of_pair.status = STARTED
                    first_of_pair.setAutoDraw(True)
                
                # if first_of_pair is active this frame...
                if first_of_pair.status == STARTED:
                    # update params
                    pass
                
                # if first_of_pair is stopping this frame...
                if first_of_pair.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > first_of_pair.tStartRefresh + 1.5-frameTolerance:
                        # keep track of stop time/frame for later
                        first_of_pair.tStop = t  # not accounting for scr refresh
                        first_of_pair.tStopRefresh = tThisFlipGlobal  # on global time
                        first_of_pair.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'first_of_pair.stopped')
                        # update status
                        first_of_pair.status = FINISHED
                        first_of_pair.setAutoDraw(False)
                
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
                    image_learn_1.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in image_learn_1.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "image_learn_1" ---
            for thisComponent in image_learn_1.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for image_learn_1
            image_learn_1.tStop = globalClock.getTime(format='float')
            image_learn_1.tStopRefresh = tThisFlipGlobal
            thisExp.addData('image_learn_1.stopped', image_learn_1.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if image_learn_1.maxDurationReached:
                routineTimer.addTime(-image_learn_1.maxDuration)
            elif image_learn_1.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.500000)
            
            # --- Prepare to start Routine "break_img_learn" ---
            # create an object to store info about Routine break_img_learn
            break_img_learn = data.Routine(
                name='break_img_learn',
                components=[break_random],
            )
            break_img_learn.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from random_break_timer
            import random 
            
            random_break = random.uniform(1.0, 3.0)
            # store start times for break_img_learn
            break_img_learn.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            break_img_learn.tStart = globalClock.getTime(format='float')
            break_img_learn.status = STARTED
            thisExp.addData('break_img_learn.started', break_img_learn.tStart)
            break_img_learn.maxDuration = None
            # keep track of which components have finished
            break_img_learnComponents = break_img_learn.components
            for thisComponent in break_img_learn.components:
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
            
            # --- Run Routine "break_img_learn" ---
            # if trial has changed, end Routine now
            if isinstance(learn_images, data.TrialHandler2) and thisLearn_image.thisN != learn_images.thisTrial.thisN:
                continueRoutine = False
            break_img_learn.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *break_random* updates
                
                # if break_random is starting this frame...
                if break_random.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    break_random.frameNStart = frameN  # exact frame index
                    break_random.tStart = t  # local t and not account for scr refresh
                    break_random.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(break_random, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'break_random.started')
                    # update status
                    break_random.status = STARTED
                    break_random.setAutoDraw(True)
                
                # if break_random is active this frame...
                if break_random.status == STARTED:
                    # update params
                    pass
                
                # if break_random is stopping this frame...
                if break_random.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > break_random.tStartRefresh + random_break-frameTolerance:
                        # keep track of stop time/frame for later
                        break_random.tStop = t  # not accounting for scr refresh
                        break_random.tStopRefresh = tThisFlipGlobal  # on global time
                        break_random.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'break_random.stopped')
                        # update status
                        break_random.status = FINISHED
                        break_random.setAutoDraw(False)
                
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
                    break_img_learn.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in break_img_learn.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "break_img_learn" ---
            for thisComponent in break_img_learn.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for break_img_learn
            break_img_learn.tStop = globalClock.getTime(format='float')
            break_img_learn.tStopRefresh = tThisFlipGlobal
            thisExp.addData('break_img_learn.stopped', break_img_learn.tStop)
            # the Routine "break_img_learn" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "image_learn_2" ---
            # create an object to store info about Routine image_learn_2
            image_learn_2 = data.Routine(
                name='image_learn_2',
                components=[second_of_pair],
            )
            image_learn_2.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for image_learn_2
            image_learn_2.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            image_learn_2.tStart = globalClock.getTime(format='float')
            image_learn_2.status = STARTED
            thisExp.addData('image_learn_2.started', image_learn_2.tStart)
            image_learn_2.maxDuration = None
            # keep track of which components have finished
            image_learn_2Components = image_learn_2.components
            for thisComponent in image_learn_2.components:
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
            
            # --- Run Routine "image_learn_2" ---
            # if trial has changed, end Routine now
            if isinstance(learn_images, data.TrialHandler2) and thisLearn_image.thisN != learn_images.thisTrial.thisN:
                continueRoutine = False
            image_learn_2.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 1.5:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *second_of_pair* updates
                
                # if second_of_pair is starting this frame...
                if second_of_pair.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    second_of_pair.frameNStart = frameN  # exact frame index
                    second_of_pair.tStart = t  # local t and not account for scr refresh
                    second_of_pair.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(second_of_pair, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'second_of_pair.started')
                    # update status
                    second_of_pair.status = STARTED
                    second_of_pair.setAutoDraw(True)
                
                # if second_of_pair is active this frame...
                if second_of_pair.status == STARTED:
                    # update params
                    pass
                
                # if second_of_pair is stopping this frame...
                if second_of_pair.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > second_of_pair.tStartRefresh + 1.5-frameTolerance:
                        # keep track of stop time/frame for later
                        second_of_pair.tStop = t  # not accounting for scr refresh
                        second_of_pair.tStopRefresh = tThisFlipGlobal  # on global time
                        second_of_pair.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'second_of_pair.stopped')
                        # update status
                        second_of_pair.status = FINISHED
                        second_of_pair.setAutoDraw(False)
                
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
                    image_learn_2.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in image_learn_2.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "image_learn_2" ---
            for thisComponent in image_learn_2.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for image_learn_2
            image_learn_2.tStop = globalClock.getTime(format='float')
            image_learn_2.tStopRefresh = tThisFlipGlobal
            thisExp.addData('image_learn_2.stopped', image_learn_2.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if image_learn_2.maxDurationReached:
                routineTimer.addTime(-image_learn_2.maxDuration)
            elif image_learn_2.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-1.500000)
            
            # --- Prepare to start Routine "routine_5s_break" ---
            # create an object to store info about Routine routine_5s_break
            routine_5s_break = data.Routine(
                name='routine_5s_break',
                components=[five_s_break],
            )
            routine_5s_break.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for routine_5s_break
            routine_5s_break.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            routine_5s_break.tStart = globalClock.getTime(format='float')
            routine_5s_break.status = STARTED
            thisExp.addData('routine_5s_break.started', routine_5s_break.tStart)
            routine_5s_break.maxDuration = None
            # keep track of which components have finished
            routine_5s_breakComponents = routine_5s_break.components
            for thisComponent in routine_5s_break.components:
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
            
            # --- Run Routine "routine_5s_break" ---
            # if trial has changed, end Routine now
            if isinstance(learn_images, data.TrialHandler2) and thisLearn_image.thisN != learn_images.thisTrial.thisN:
                continueRoutine = False
            routine_5s_break.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 5.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *five_s_break* updates
                
                # if five_s_break is starting this frame...
                if five_s_break.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    five_s_break.frameNStart = frameN  # exact frame index
                    five_s_break.tStart = t  # local t and not account for scr refresh
                    five_s_break.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(five_s_break, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'five_s_break.started')
                    # update status
                    five_s_break.status = STARTED
                    five_s_break.setAutoDraw(True)
                
                # if five_s_break is active this frame...
                if five_s_break.status == STARTED:
                    # update params
                    pass
                
                # if five_s_break is stopping this frame...
                if five_s_break.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > five_s_break.tStartRefresh + 5-frameTolerance:
                        # keep track of stop time/frame for later
                        five_s_break.tStop = t  # not accounting for scr refresh
                        five_s_break.tStopRefresh = tThisFlipGlobal  # on global time
                        five_s_break.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'five_s_break.stopped')
                        # update status
                        five_s_break.status = FINISHED
                        five_s_break.setAutoDraw(False)
                # Run 'Each Frame' code from save_data_learn
                if fix_learn_cross.status == STARTED and wait_sent == 0: #If the stimulus component has started and the trigger has not yet been sent. Change 'stimulus' to match the name of the component you want the trigger to be sent at the same time as
                    win.callOnFlip(port.write, data=b'\x20') #Send the trigger, synced to the screen refresh
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
                    routine_5s_break.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in routine_5s_break.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "routine_5s_break" ---
            for thisComponent in routine_5s_break.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for routine_5s_break
            routine_5s_break.tStop = globalClock.getTime(format='float')
            routine_5s_break.tStopRefresh = tThisFlipGlobal
            thisExp.addData('routine_5s_break.stopped', routine_5s_break.tStop)
            # Run 'End Routine' code from save_data_learn
            # store data
            thisExp.addData('pair_stim_1',stim1_img)
            thisExp.addData('pair_stim_2',stim2_img)
            thisExp.addData('random_break_dur',random_break)
            
            
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if routine_5s_break.maxDurationReached:
                routineTimer.addTime(-routine_5s_break.maxDuration)
            elif routine_5s_break.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-5.000000)
            thisExp.nextEntry()
            
        # completed 3.0 repeats of 'learn_images'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # get names of stimulus parameters
        if learn_images.trialList in ([], [None], None):
            params = []
        else:
            params = learn_images.trialList[0].keys()
        # save data for this loop
        learn_images.saveAsText(filename + 'learn_images.csv', delim=',',
            stimOut=params,
            dataOut=['n','all_mean','all_std', 'all_raw'])
        
        # --- Prepare to start Routine "probe_instr" ---
        # create an object to store info about Routine probe_instr
        probe_instr = data.Routine(
            name='probe_instr',
            components=[probe_instr_text, end_probe_instr],
        )
        probe_instr.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for end_probe_instr
        end_probe_instr.keys = []
        end_probe_instr.rt = []
        _end_probe_instr_allKeys = []
        # store start times for probe_instr
        probe_instr.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        probe_instr.tStart = globalClock.getTime(format='float')
        probe_instr.status = STARTED
        thisExp.addData('probe_instr.started', probe_instr.tStart)
        probe_instr.maxDuration = None
        # keep track of which components have finished
        probe_instrComponents = probe_instr.components
        for thisComponent in probe_instr.components:
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
        
        # --- Run Routine "probe_instr" ---
        # if trial has changed, end Routine now
        if isinstance(blocks_loop, data.TrialHandler2) and thisBlocks_loop.thisN != blocks_loop.thisTrial.thisN:
            continueRoutine = False
        probe_instr.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *probe_instr_text* updates
            
            # if probe_instr_text is starting this frame...
            if probe_instr_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                probe_instr_text.frameNStart = frameN  # exact frame index
                probe_instr_text.tStart = t  # local t and not account for scr refresh
                probe_instr_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(probe_instr_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'probe_instr_text.started')
                # update status
                probe_instr_text.status = STARTED
                probe_instr_text.setAutoDraw(True)
            
            # if probe_instr_text is active this frame...
            if probe_instr_text.status == STARTED:
                # update params
                pass
            
            # *end_probe_instr* updates
            waitOnFlip = False
            
            # if end_probe_instr is starting this frame...
            if end_probe_instr.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                end_probe_instr.frameNStart = frameN  # exact frame index
                end_probe_instr.tStart = t  # local t and not account for scr refresh
                end_probe_instr.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(end_probe_instr, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'end_probe_instr.started')
                # update status
                end_probe_instr.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(end_probe_instr.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(end_probe_instr.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if end_probe_instr.status == STARTED and not waitOnFlip:
                theseKeys = end_probe_instr.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _end_probe_instr_allKeys.extend(theseKeys)
                if len(_end_probe_instr_allKeys):
                    end_probe_instr.keys = _end_probe_instr_allKeys[-1].name  # just the last key pressed
                    end_probe_instr.rt = _end_probe_instr_allKeys[-1].rt
                    end_probe_instr.duration = _end_probe_instr_allKeys[-1].duration
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
                probe_instr.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in probe_instr.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "probe_instr" ---
        for thisComponent in probe_instr.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for probe_instr
        probe_instr.tStop = globalClock.getTime(format='float')
        probe_instr.tStopRefresh = tThisFlipGlobal
        thisExp.addData('probe_instr.stopped', probe_instr.tStop)
        # check responses
        if end_probe_instr.keys in ['', [], None]:  # No response was made
            end_probe_instr.keys = None
        blocks_loop.addData('end_probe_instr.keys',end_probe_instr.keys)
        if end_probe_instr.keys != None:  # we had a response
            blocks_loop.addData('end_probe_instr.rt', end_probe_instr.rt)
            blocks_loop.addData('end_probe_instr.duration', end_probe_instr.duration)
        # the Routine "probe_instr" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # set up handler to look after randomisation of conditions etc
        probe_loop = data.TrialHandler2(
            name='probe_loop',
            nReps=1.0, 
            method='random', 
            extraInfo=expInfo, 
            originPath=-1, 
            trialList=data.importConditions(condition_probe), 
            seed=None, 
        )
        thisExp.addLoop(probe_loop)  # add the loop to the experiment
        thisProbe_loop = probe_loop.trialList[0]  # so we can initialise stimuli with some values
        # abbreviate parameter names if possible (e.g. rgb = thisProbe_loop.rgb)
        if thisProbe_loop != None:
            for paramName in thisProbe_loop:
                globals()[paramName] = thisProbe_loop[paramName]
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        
        for thisProbe_loop in probe_loop:
            currentLoop = probe_loop
            thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
            if thisSession is not None:
                # if running in a Session with a Liaison client, send data up to now
                thisSession.sendExperimentData()
            # abbreviate parameter names if possible (e.g. rgb = thisProbe_loop.rgb)
            if thisProbe_loop != None:
                for paramName in thisProbe_loop:
                    globals()[paramName] = thisProbe_loop[paramName]
            
            # --- Prepare to start Routine "image_loader" ---
            # create an object to store info about Routine image_loader
            image_loader = data.Routine(
                name='image_loader',
                components=[empty_text, image_load],
            )
            image_loader.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for image_loader
            image_loader.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            image_loader.tStart = globalClock.getTime(format='float')
            image_loader.status = STARTED
            thisExp.addData('image_loader.started', image_loader.tStart)
            image_loader.maxDuration = None
            # keep track of which components have finished
            image_loaderComponents = image_loader.components
            for thisComponent in image_loader.components:
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
            
            # --- Run Routine "image_loader" ---
            # if trial has changed, end Routine now
            if isinstance(probe_loop, data.TrialHandler2) and thisProbe_loop.thisN != probe_loop.thisTrial.thisN:
                continueRoutine = False
            image_loader.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 0.3:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *empty_text* updates
                
                # if empty_text is starting this frame...
                if empty_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    empty_text.frameNStart = frameN  # exact frame index
                    empty_text.tStart = t  # local t and not account for scr refresh
                    empty_text.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(empty_text, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'empty_text.started')
                    # update status
                    empty_text.status = STARTED
                    empty_text.setAutoDraw(True)
                
                # if empty_text is active this frame...
                if empty_text.status == STARTED:
                    # update params
                    pass
                
                # if empty_text is stopping this frame...
                if empty_text.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > empty_text.tStartRefresh + 0.3-frameTolerance:
                        # keep track of stop time/frame for later
                        empty_text.tStop = t  # not accounting for scr refresh
                        empty_text.tStopRefresh = tThisFlipGlobal  # on global time
                        empty_text.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'empty_text.stopped')
                        # update status
                        empty_text.status = FINISHED
                        empty_text.setAutoDraw(False)
                # *image_load* period
                
                # if image_load is starting this frame...
                if image_load.status == NOT_STARTED and t >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    image_load.frameNStart = frameN  # exact frame index
                    image_load.tStart = t  # local t and not account for scr refresh
                    image_load.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(image_load, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.addData('image_load.started', t)
                    # update status
                    image_load.status = STARTED
                    image_load.start(0.3)
                elif image_load.status == STARTED:  # one frame should pass before updating params and completing
                    # Updating other components during *image_load*
                    probe_cue_1.setImage(target_img)
                    prob_test_img.setImage(prob_img)
                    # Component updates done
                    image_load.complete()  # finish the static period
                    image_load.tStop = image_load.tStart + 0.3  # record stop time
                
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
                    image_loader.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in image_loader.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "image_loader" ---
            for thisComponent in image_loader.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for image_loader
            image_loader.tStop = globalClock.getTime(format='float')
            image_loader.tStopRefresh = tThisFlipGlobal
            thisExp.addData('image_loader.stopped', image_loader.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if image_loader.maxDurationReached:
                routineTimer.addTime(-image_loader.maxDuration)
            elif image_loader.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-0.300000)
            
            # --- Prepare to start Routine "probe_cue" ---
            # create an object to store info about Routine probe_cue
            probe_cue = data.Routine(
                name='probe_cue',
                components=[probe_cue_1],
            )
            probe_cue.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for probe_cue
            probe_cue.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            probe_cue.tStart = globalClock.getTime(format='float')
            probe_cue.status = STARTED
            thisExp.addData('probe_cue.started', probe_cue.tStart)
            probe_cue.maxDuration = None
            # keep track of which components have finished
            probe_cueComponents = probe_cue.components
            for thisComponent in probe_cue.components:
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
            
            # --- Run Routine "probe_cue" ---
            # if trial has changed, end Routine now
            if isinstance(probe_loop, data.TrialHandler2) and thisProbe_loop.thisN != probe_loop.thisTrial.thisN:
                continueRoutine = False
            probe_cue.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 4.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *probe_cue_1* updates
                
                # if probe_cue_1 is starting this frame...
                if probe_cue_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    probe_cue_1.frameNStart = frameN  # exact frame index
                    probe_cue_1.tStart = t  # local t and not account for scr refresh
                    probe_cue_1.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(probe_cue_1, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'probe_cue_1.started')
                    # update status
                    probe_cue_1.status = STARTED
                    probe_cue_1.setAutoDraw(True)
                
                # if probe_cue_1 is active this frame...
                if probe_cue_1.status == STARTED:
                    # update params
                    pass
                
                # if probe_cue_1 is stopping this frame...
                if probe_cue_1.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > probe_cue_1.tStartRefresh + 4-frameTolerance:
                        # keep track of stop time/frame for later
                        probe_cue_1.tStop = t  # not accounting for scr refresh
                        probe_cue_1.tStopRefresh = tThisFlipGlobal  # on global time
                        probe_cue_1.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'probe_cue_1.stopped')
                        # update status
                        probe_cue_1.status = FINISHED
                        probe_cue_1.setAutoDraw(False)
                
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
                    probe_cue.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in probe_cue.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "probe_cue" ---
            for thisComponent in probe_cue.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for probe_cue
            probe_cue.tStop = globalClock.getTime(format='float')
            probe_cue.tStopRefresh = tThisFlipGlobal
            thisExp.addData('probe_cue.stopped', probe_cue.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if probe_cue.maxDurationReached:
                routineTimer.addTime(-probe_cue.maxDuration)
            elif probe_cue.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-4.000000)
            
            # --- Prepare to start Routine "probe_imagination" ---
            # create an object to store info about Routine probe_imagination
            probe_imagination = data.Routine(
                name='probe_imagination',
                components=[imagination_woot],
            )
            probe_imagination.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from imagination_timer
            import random 
            
            prob_imagination_dur = random.uniform(1.0, 3.0)
            imagination_woot.setText('')
            # store start times for probe_imagination
            probe_imagination.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            probe_imagination.tStart = globalClock.getTime(format='float')
            probe_imagination.status = STARTED
            thisExp.addData('probe_imagination.started', probe_imagination.tStart)
            probe_imagination.maxDuration = None
            # keep track of which components have finished
            probe_imaginationComponents = probe_imagination.components
            for thisComponent in probe_imagination.components:
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
            
            # --- Run Routine "probe_imagination" ---
            # if trial has changed, end Routine now
            if isinstance(probe_loop, data.TrialHandler2) and thisProbe_loop.thisN != probe_loop.thisTrial.thisN:
                continueRoutine = False
            probe_imagination.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *imagination_woot* updates
                
                # if imagination_woot is starting this frame...
                if imagination_woot.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    imagination_woot.frameNStart = frameN  # exact frame index
                    imagination_woot.tStart = t  # local t and not account for scr refresh
                    imagination_woot.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(imagination_woot, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'imagination_woot.started')
                    # update status
                    imagination_woot.status = STARTED
                    imagination_woot.setAutoDraw(True)
                
                # if imagination_woot is active this frame...
                if imagination_woot.status == STARTED:
                    # update params
                    pass
                
                # if imagination_woot is stopping this frame...
                if imagination_woot.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > imagination_woot.tStartRefresh + prob_imagination_dur-frameTolerance:
                        # keep track of stop time/frame for later
                        imagination_woot.tStop = t  # not accounting for scr refresh
                        imagination_woot.tStopRefresh = tThisFlipGlobal  # on global time
                        imagination_woot.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'imagination_woot.stopped')
                        # update status
                        imagination_woot.status = FINISHED
                        imagination_woot.setAutoDraw(False)
                
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
                    probe_imagination.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in probe_imagination.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "probe_imagination" ---
            for thisComponent in probe_imagination.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for probe_imagination
            probe_imagination.tStop = globalClock.getTime(format='float')
            probe_imagination.tStopRefresh = tThisFlipGlobal
            thisExp.addData('probe_imagination.stopped', probe_imagination.tStop)
            # the Routine "probe_imagination" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            
            # --- Prepare to start Routine "probe_test" ---
            # create an object to store info about Routine probe_test
            probe_test = data.Routine(
                name='probe_test',
                components=[prob_test_img],
            )
            probe_test.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # store start times for probe_test
            probe_test.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            probe_test.tStart = globalClock.getTime(format='float')
            probe_test.status = STARTED
            thisExp.addData('probe_test.started', probe_test.tStart)
            probe_test.maxDuration = None
            # keep track of which components have finished
            probe_testComponents = probe_test.components
            for thisComponent in probe_test.components:
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
            
            # --- Run Routine "probe_test" ---
            # if trial has changed, end Routine now
            if isinstance(probe_loop, data.TrialHandler2) and thisProbe_loop.thisN != probe_loop.thisTrial.thisN:
                continueRoutine = False
            probe_test.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 2.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *prob_test_img* updates
                
                # if prob_test_img is starting this frame...
                if prob_test_img.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    prob_test_img.frameNStart = frameN  # exact frame index
                    prob_test_img.tStart = t  # local t and not account for scr refresh
                    prob_test_img.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(prob_test_img, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'prob_test_img.started')
                    # update status
                    prob_test_img.status = STARTED
                    prob_test_img.setAutoDraw(True)
                
                # if prob_test_img is active this frame...
                if prob_test_img.status == STARTED:
                    # update params
                    pass
                
                # if prob_test_img is stopping this frame...
                if prob_test_img.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > prob_test_img.tStartRefresh + 2-frameTolerance:
                        # keep track of stop time/frame for later
                        prob_test_img.tStop = t  # not accounting for scr refresh
                        prob_test_img.tStopRefresh = tThisFlipGlobal  # on global time
                        prob_test_img.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'prob_test_img.stopped')
                        # update status
                        prob_test_img.status = FINISHED
                        prob_test_img.setAutoDraw(False)
                
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
                    probe_test.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in probe_test.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "probe_test" ---
            for thisComponent in probe_test.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for probe_test
            probe_test.tStop = globalClock.getTime(format='float')
            probe_test.tStopRefresh = tThisFlipGlobal
            thisExp.addData('probe_test.stopped', probe_test.tStop)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if probe_test.maxDurationReached:
                routineTimer.addTime(-probe_test.maxDuration)
            elif probe_test.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-2.000000)
            
            # --- Prepare to start Routine "probe_resp" ---
            # create an object to store info about Routine probe_resp
            probe_resp = data.Routine(
                name='probe_resp',
                components=[probe_resp_test, text_4],
            )
            probe_resp.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # create starting attributes for probe_resp_test
            probe_resp_test.keys = []
            probe_resp_test.rt = []
            _probe_resp_test_allKeys = []
            # store start times for probe_resp
            probe_resp.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            probe_resp.tStart = globalClock.getTime(format='float')
            probe_resp.status = STARTED
            thisExp.addData('probe_resp.started', probe_resp.tStart)
            probe_resp.maxDuration = None
            # keep track of which components have finished
            probe_respComponents = probe_resp.components
            for thisComponent in probe_resp.components:
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
            
            # --- Run Routine "probe_resp" ---
            # if trial has changed, end Routine now
            if isinstance(probe_loop, data.TrialHandler2) and thisProbe_loop.thisN != probe_loop.thisTrial.thisN:
                continueRoutine = False
            probe_resp.forceEnded = routineForceEnded = not continueRoutine
            while continueRoutine and routineTimer.getTime() < 4.0:
                # get current time
                t = routineTimer.getTime()
                tThisFlip = win.getFutureFlipTime(clock=routineTimer)
                tThisFlipGlobal = win.getFutureFlipTime(clock=None)
                frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
                # update/draw components on each frame
                
                # *probe_resp_test* updates
                waitOnFlip = False
                
                # if probe_resp_test is starting this frame...
                if probe_resp_test.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    probe_resp_test.frameNStart = frameN  # exact frame index
                    probe_resp_test.tStart = t  # local t and not account for scr refresh
                    probe_resp_test.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(probe_resp_test, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'probe_resp_test.started')
                    # update status
                    probe_resp_test.status = STARTED
                    # keyboard checking is just starting
                    waitOnFlip = True
                    win.callOnFlip(probe_resp_test.clock.reset)  # t=0 on next screen flip
                    win.callOnFlip(probe_resp_test.clearEvents, eventType='keyboard')  # clear events on next screen flip
                
                # if probe_resp_test is stopping this frame...
                if probe_resp_test.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > probe_resp_test.tStartRefresh + 4-frameTolerance:
                        # keep track of stop time/frame for later
                        probe_resp_test.tStop = t  # not accounting for scr refresh
                        probe_resp_test.tStopRefresh = tThisFlipGlobal  # on global time
                        probe_resp_test.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'probe_resp_test.stopped')
                        # update status
                        probe_resp_test.status = FINISHED
                        probe_resp_test.status = FINISHED
                if probe_resp_test.status == STARTED and not waitOnFlip:
                    theseKeys = probe_resp_test.getKeys(keyList=['1','2'], ignoreKeys=["escape"], waitRelease=False)
                    _probe_resp_test_allKeys.extend(theseKeys)
                    if len(_probe_resp_test_allKeys):
                        probe_resp_test.keys = _probe_resp_test_allKeys[-1].name  # just the last key pressed
                        probe_resp_test.rt = _probe_resp_test_allKeys[-1].rt
                        probe_resp_test.duration = _probe_resp_test_allKeys[-1].duration
                        # a response ends the routine
                        continueRoutine = False
                
                # *text_4* updates
                
                # if text_4 is starting this frame...
                if text_4.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                    # keep track of start time/frame for later
                    text_4.frameNStart = frameN  # exact frame index
                    text_4.tStart = t  # local t and not account for scr refresh
                    text_4.tStartRefresh = tThisFlipGlobal  # on global time
                    win.timeOnFlip(text_4, 'tStartRefresh')  # time at next scr refresh
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'text_4.started')
                    # update status
                    text_4.status = STARTED
                    text_4.setAutoDraw(True)
                
                # if text_4 is active this frame...
                if text_4.status == STARTED:
                    # update params
                    pass
                
                # if text_4 is stopping this frame...
                if text_4.status == STARTED:
                    # is it time to stop? (based on global clock, using actual start)
                    if tThisFlipGlobal > text_4.tStartRefresh + 4-frameTolerance:
                        # keep track of stop time/frame for later
                        text_4.tStop = t  # not accounting for scr refresh
                        text_4.tStopRefresh = tThisFlipGlobal  # on global time
                        text_4.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_4.stopped')
                        # update status
                        text_4.status = FINISHED
                        text_4.setAutoDraw(False)
                
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
                    probe_resp.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in probe_resp.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "probe_resp" ---
            for thisComponent in probe_resp.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for probe_resp
            probe_resp.tStop = globalClock.getTime(format='float')
            probe_resp.tStopRefresh = tThisFlipGlobal
            thisExp.addData('probe_resp.stopped', probe_resp.tStop)
            # check responses
            if probe_resp_test.keys in ['', [], None]:  # No response was made
                probe_resp_test.keys = None
            probe_loop.addData('probe_resp_test.keys',probe_resp_test.keys)
            if probe_resp_test.keys != None:  # we had a response
                probe_loop.addData('probe_resp_test.rt', probe_resp_test.rt)
                probe_loop.addData('probe_resp_test.duration', probe_resp_test.duration)
            # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
            if probe_resp.maxDurationReached:
                routineTimer.addTime(-probe_resp.maxDuration)
            elif probe_resp.forceEnded:
                routineTimer.reset()
            else:
                routineTimer.addTime(-4.000000)
            
            # --- Prepare to start Routine "probe_break" ---
            # create an object to store info about Routine probe_break
            probe_break = data.Routine(
                name='probe_break',
                components=[text_2],
            )
            probe_break.status = NOT_STARTED
            continueRoutine = True
            # update component parameters for each repeat
            # Run 'Begin Routine' code from code
            import random 
            
            prob_break = random.uniform(1.0, 3.0)
            text_2.setText('')
            # store start times for probe_break
            probe_break.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
            probe_break.tStart = globalClock.getTime(format='float')
            probe_break.status = STARTED
            thisExp.addData('probe_break.started', probe_break.tStart)
            probe_break.maxDuration = None
            # keep track of which components have finished
            probe_breakComponents = probe_break.components
            for thisComponent in probe_break.components:
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
            
            # --- Run Routine "probe_break" ---
            # if trial has changed, end Routine now
            if isinstance(probe_loop, data.TrialHandler2) and thisProbe_loop.thisN != probe_loop.thisTrial.thisN:
                continueRoutine = False
            probe_break.forceEnded = routineForceEnded = not continueRoutine
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
                    if tThisFlipGlobal > text_2.tStartRefresh + prob_break-frameTolerance:
                        # keep track of stop time/frame for later
                        text_2.tStop = t  # not accounting for scr refresh
                        text_2.tStopRefresh = tThisFlipGlobal  # on global time
                        text_2.frameNStop = frameN  # exact frame index
                        # add timestamp to datafile
                        thisExp.timestampOnFlip(win, 'text_2.stopped')
                        # update status
                        text_2.status = FINISHED
                        text_2.setAutoDraw(False)
                
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
                    probe_break.forceEnded = routineForceEnded = True
                    break
                continueRoutine = False  # will revert to True if at least one component still running
                for thisComponent in probe_break.components:
                    if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                        continueRoutine = True
                        break  # at least one component has not yet finished
                
                # refresh the screen
                if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                    win.flip()
            
            # --- Ending Routine "probe_break" ---
            for thisComponent in probe_break.components:
                if hasattr(thisComponent, "setAutoDraw"):
                    thisComponent.setAutoDraw(False)
            # store stop times for probe_break
            probe_break.tStop = globalClock.getTime(format='float')
            probe_break.tStopRefresh = tThisFlipGlobal
            thisExp.addData('probe_break.stopped', probe_break.tStop)
            # Run 'End Routine' code from save_data
            #store data
            thisExp.addData('target_image', target_img)
            thisExp.addData('probe_image', prob_img)
            
            if is_match == 'match':
                correct_key = sequence_good
            else :
                correct_key = sequence_bad
                
            if probe_resp_test.keys and probe_resp_test.keys[0] == correct_key:
                thisExp.addData('response_good_bad', 1)
                answer_counter = answer_counter + 1
            else:
                thisExp.addData('response_good_bad', 0)
                
            if probe_resp_test.keys:
                thisExp.addData('response_given', probe_resp_test.keys[0])
                thisExp.addData('response_rt', key_resp.rt)
            else: 
                thisExp.addData('response_given', None) 
                thisExp.addData('response_rt', None)
                
            thisExp.addData('good_answer_counter', answer_counter)
            # the Routine "probe_break" was not non-slip safe, so reset the non-slip timer
            routineTimer.reset()
            thisExp.nextEntry()
            
        # completed 1.0 repeats of 'probe_loop'
        
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # get names of stimulus parameters
        if probe_loop.trialList in ([], [None], None):
            params = []
        else:
            params = probe_loop.trialList[0].keys()
        # save data for this loop
        probe_loop.saveAsText(filename + 'probe_loop.csv', delim=',',
            stimOut=params,
            dataOut=['n','all_mean','all_std', 'all_raw'])
        
        # --- Prepare to start Routine "end_of_block" ---
        # create an object to store info about Routine end_of_block
        end_of_block = data.Routine(
            name='end_of_block',
            components=[text_3, key_resp_2],
        )
        end_of_block.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # create starting attributes for key_resp_2
        key_resp_2.keys = []
        key_resp_2.rt = []
        _key_resp_2_allKeys = []
        # store start times for end_of_block
        end_of_block.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        end_of_block.tStart = globalClock.getTime(format='float')
        end_of_block.status = STARTED
        thisExp.addData('end_of_block.started', end_of_block.tStart)
        end_of_block.maxDuration = None
        # keep track of which components have finished
        end_of_blockComponents = end_of_block.components
        for thisComponent in end_of_block.components:
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
        
        # --- Run Routine "end_of_block" ---
        # if trial has changed, end Routine now
        if isinstance(blocks_loop, data.TrialHandler2) and thisBlocks_loop.thisN != blocks_loop.thisTrial.thisN:
            continueRoutine = False
        end_of_block.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
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
                end_of_block.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in end_of_block.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "end_of_block" ---
        for thisComponent in end_of_block.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for end_of_block
        end_of_block.tStop = globalClock.getTime(format='float')
        end_of_block.tStopRefresh = tThisFlipGlobal
        thisExp.addData('end_of_block.stopped', end_of_block.tStop)
        # check responses
        if key_resp_2.keys in ['', [], None]:  # No response was made
            key_resp_2.keys = None
        blocks_loop.addData('key_resp_2.keys',key_resp_2.keys)
        if key_resp_2.keys != None:  # we had a response
            blocks_loop.addData('key_resp_2.rt', key_resp_2.rt)
            blocks_loop.addData('key_resp_2.duration', key_resp_2.duration)
        # the Routine "end_of_block" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 3.0 repeats of 'blocks_loop'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    # get names of stimulus parameters
    if blocks_loop.trialList in ([], [None], None):
        params = []
    else:
        params = blocks_loop.trialList[0].keys()
    # save data for this loop
    blocks_loop.saveAsText(filename + 'blocks_loop.csv', delim=',',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    
    # --- Prepare to start Routine "end_exp" ---
    # create an object to store info about Routine end_exp
    end_exp = data.Routine(
        name='end_exp',
        components=[text_end, end_exp_1],
    )
    end_exp.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for end_exp_1
    end_exp_1.keys = []
    end_exp_1.rt = []
    _end_exp_1_allKeys = []
    # store start times for end_exp
    end_exp.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    end_exp.tStart = globalClock.getTime(format='float')
    end_exp.status = STARTED
    thisExp.addData('end_exp.started', end_exp.tStart)
    end_exp.maxDuration = None
    # keep track of which components have finished
    end_expComponents = end_exp.components
    for thisComponent in end_exp.components:
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
    
    # --- Run Routine "end_exp" ---
    end_exp.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_end* updates
        
        # if text_end is starting this frame...
        if text_end.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_end.frameNStart = frameN  # exact frame index
            text_end.tStart = t  # local t and not account for scr refresh
            text_end.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_end, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_end.started')
            # update status
            text_end.status = STARTED
            text_end.setAutoDraw(True)
        
        # if text_end is active this frame...
        if text_end.status == STARTED:
            # update params
            pass
        
        # if text_end is stopping this frame...
        if text_end.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > text_end.tStartRefresh + 20-frameTolerance:
                # keep track of stop time/frame for later
                text_end.tStop = t  # not accounting for scr refresh
                text_end.tStopRefresh = tThisFlipGlobal  # on global time
                text_end.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_end.stopped')
                # update status
                text_end.status = FINISHED
                text_end.setAutoDraw(False)
        
        # *end_exp_1* updates
        waitOnFlip = False
        
        # if end_exp_1 is starting this frame...
        if end_exp_1.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            end_exp_1.frameNStart = frameN  # exact frame index
            end_exp_1.tStart = t  # local t and not account for scr refresh
            end_exp_1.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(end_exp_1, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'end_exp_1.started')
            # update status
            end_exp_1.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(end_exp_1.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(end_exp_1.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if end_exp_1.status == STARTED and not waitOnFlip:
            theseKeys = end_exp_1.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _end_exp_1_allKeys.extend(theseKeys)
            if len(_end_exp_1_allKeys):
                end_exp_1.keys = _end_exp_1_allKeys[-1].name  # just the last key pressed
                end_exp_1.rt = _end_exp_1_allKeys[-1].rt
                end_exp_1.duration = _end_exp_1_allKeys[-1].duration
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
            end_exp.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in end_exp.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end_exp" ---
    for thisComponent in end_exp.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for end_exp
    end_exp.tStop = globalClock.getTime(format='float')
    end_exp.tStopRefresh = tThisFlipGlobal
    thisExp.addData('end_exp.stopped', end_exp.tStop)
    # check responses
    if end_exp_1.keys in ['', [], None]:  # No response was made
        end_exp_1.keys = None
    thisExp.addData('end_exp_1.keys',end_exp_1.keys)
    if end_exp_1.keys != None:  # we had a response
        thisExp.addData('end_exp_1.rt', end_exp_1.rt)
        thisExp.addData('end_exp_1.duration', end_exp_1.duration)
    thisExp.nextEntry()
    # the Routine "end_exp" was not non-slip safe, so reset the non-slip timer
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
