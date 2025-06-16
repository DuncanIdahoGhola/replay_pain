#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2024.2.4),
    on juin 16, 2025, at 10:30
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
expName = 'cued_mental_stim'  # from the Builder filename that created this script
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
    filename = u'data/%s/cued_mental_stim/%s_%s_%s' % (expInfo['participant'], expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version='',
        extraInfo=expInfo, runtimeInfo=None,
        originPath='C:\\Users\\cbant\\OneDrive\\Bureau\\git\\replay_pain\\replay_pilot\\3.cued_mental_stim_lastrun.py',
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
    if deviceManager.getDevice('quit_instr') is None:
        # initialise quit_instr
        quit_instr = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='quit_instr',
        )
    if deviceManager.getDevice('exit_eeg_text') is None:
        # initialise exit_eeg_text
        exit_eeg_text = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='exit_eeg_text',
        )
    if deviceManager.getDevice('key_resp_probe') is None:
        # initialise key_resp_probe
        key_resp_probe = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_probe',
        )
    if deviceManager.getDevice('key_resp_rating') is None:
        # initialise key_resp_rating
        key_resp_rating = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_rating',
        )
    if deviceManager.getDevice('key_resp_break') is None:
        # initialise key_resp_break
        key_resp_break = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='key_resp_break',
        )
    if deviceManager.getDevice('exit_end') is None:
        # initialise exit_end
        exit_end = deviceManager.addDevice(
            deviceClass='keyboard',
            deviceName='exit_end',
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
    
    # --- Initialize components for Routine "instr_task" ---
    # Run 'Begin Experiment' code from start_code
    import random
    import pandas as pd
    #add counterbalance and allowed keys
    participant_id = expInfo['participant']
    last_three_digits = int(participant_id[-3:])
    
    
    
    instr_1 = 'Un indice apparaîtra pour vous guider dans la mentalisation de la séquence: si vous voyez "1 - 4", imaginez les images dans cet ordre de 1 à 4 ; si "4 - 1", imaginez-les de 4 à 1. Ensuite, une image sera présenté et vous devrez décider si elle fait partie de la séquence que vous venez de visualiser mentalement : appuyez sur 1 si elle en fait partie, et sur 2 sinon. Après votre réponse, évaluez la vivacité de votre mentalisation sur une échelle de 1 à 5, où 1 signifie "Très peu vive" et 5 "Très vive". Appuyez sur la barre espace pour commencer.'
    instr_2 = 'Un indice apparaîtra pour vous guider dans la mentalisation de la séquence: si vous voyez "1 - 4", imaginez les images dans cet ordre de 1 à 4 ; si "4 - 1", imaginez-les de 4 à 1. Ensuite, une image sera présenté et vous devrez décider si elle fait partie de la séquence que vous venez de visualiser mentalement : appuyez sur 2 si elle en fait partie, et sur 1 sinon. Après votre réponse, évaluez la vivacité de votre mentalisation sur une échelle de 1 à 5, où 1 signifie "Très peu vive" et 5 "Très vive". Appuyez sur la barre espace pour commencer.'
    
    if last_three_digits % 2 == 0:
        sequence_good = '1'
        sequence_bad = '2'
        instr_message = instr_1
    else : 
        sequence_good = '2'
        sequence_bad = '1'
        instr_message = instr_2
    
    allowed_keys_list = [sequence_good, sequence_bad]
    
    # Store these for later use in routines
    expInfo['sequence_good_key'] = sequence_good
    expInfo['sequence_bad_key'] = sequence_bad
    expInfo['allowed_keys_list'] = allowed_keys_list
    
    
    condition_part = f"conditions/p{last_three_digits}_mental_sim_conditions.csv"
    
    
    task_instr_text = visual.TextStim(win=win, name='task_instr_text',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    quit_instr = keyboard.Keyboard(deviceName='quit_instr')
    
    # --- Initialize components for Routine "start_eeg" ---
    EEG_start_text = visual.TextStim(win=win, name='EEG_start_text',
        text='Check EEG recording',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    exit_eeg_text = keyboard.Keyboard(deviceName='exit_eeg_text')
    
    # --- Initialize components for Routine "image_load" ---
    image_load_text = visual.TextStim(win=win, name='image_load_text',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.15, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    imag_loader = clock.StaticPeriod(win=win, screenHz=expInfo['frameRate'], name='imag_loader')
    
    # --- Initialize components for Routine "trial" ---
    cue_stim = visual.TextStim(win=win, name='cue_stim',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    fixation_stim = visual.TextStim(win=win, name='fixation_stim',
        text='+',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.15, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-2.0);
    probe_image = visual.ImageStim(
        win=win,
        name='probe_image', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-3.0)
    key_resp_probe = keyboard.Keyboard(deviceName='key_resp_probe')
    resp_instr = visual.TextStim(win=win, name='resp_instr',
        text='',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-5.0);
    
    # --- Initialize components for Routine "rating" ---
    rating_vividness = visual.Slider(win=win, name='rating_vividness',
        startValue=None, size=(1.0, 0.1), pos=[0, -0.1], units=win.units,
        labels=['Not at all', 'Slightly', 'Moderately', 'Very'], ticks=(1, 2, 3, 4), granularity=0.0,
        style='rating', styleTweaks=(), opacity=None,
        labelColor='LightGray', markerColor='Red', lineColor='White', colorSpace='rgb',
        font='Open Sans', labelHeight=0.05,
        flip=False, ori=0.0, depth=0, readOnly=True)
    text_rating_question = visual.TextStim(win=win, name='text_rating_question',
        text='Quelle était la vivacité de votre simulation mentale ?',
        font='Arial',
        pos=[0, 0.2], draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=-1.0);
    key_resp_rating = keyboard.Keyboard(deviceName='key_resp_rating')
    
    # --- Initialize components for Routine "block_break" ---
    block_break_text = visual.TextStim(win=win, name='block_break_text',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_break = keyboard.Keyboard(deviceName='key_resp_break')
    
    # --- Initialize components for Routine "end_task" ---
    end_text = visual.TextStim(win=win, name='end_text',
        text='Thank you text ',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.05, wrapWidth=None, ori=0.0, 
        color='white', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    exit_end = keyboard.Keyboard(deviceName='exit_end')
    
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
    
    # --- Prepare to start Routine "instr_task" ---
    # create an object to store info about Routine instr_task
    instr_task = data.Routine(
        name='instr_task',
        components=[task_instr_text, quit_instr],
    )
    instr_task.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    task_instr_text.setText(instr_message)
    # create starting attributes for quit_instr
    quit_instr.keys = []
    quit_instr.rt = []
    _quit_instr_allKeys = []
    # store start times for instr_task
    instr_task.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instr_task.tStart = globalClock.getTime(format='float')
    instr_task.status = STARTED
    thisExp.addData('instr_task.started', instr_task.tStart)
    instr_task.maxDuration = None
    # keep track of which components have finished
    instr_taskComponents = instr_task.components
    for thisComponent in instr_task.components:
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
    
    # --- Run Routine "instr_task" ---
    instr_task.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *task_instr_text* updates
        
        # if task_instr_text is starting this frame...
        if task_instr_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            task_instr_text.frameNStart = frameN  # exact frame index
            task_instr_text.tStart = t  # local t and not account for scr refresh
            task_instr_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(task_instr_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'task_instr_text.started')
            # update status
            task_instr_text.status = STARTED
            task_instr_text.setAutoDraw(True)
        
        # if task_instr_text is active this frame...
        if task_instr_text.status == STARTED:
            # update params
            pass
        
        # *quit_instr* updates
        waitOnFlip = False
        
        # if quit_instr is starting this frame...
        if quit_instr.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            quit_instr.frameNStart = frameN  # exact frame index
            quit_instr.tStart = t  # local t and not account for scr refresh
            quit_instr.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(quit_instr, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'quit_instr.started')
            # update status
            quit_instr.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(quit_instr.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(quit_instr.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if quit_instr.status == STARTED and not waitOnFlip:
            theseKeys = quit_instr.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _quit_instr_allKeys.extend(theseKeys)
            if len(_quit_instr_allKeys):
                quit_instr.keys = _quit_instr_allKeys[-1].name  # just the last key pressed
                quit_instr.rt = _quit_instr_allKeys[-1].rt
                quit_instr.duration = _quit_instr_allKeys[-1].duration
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
            instr_task.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in instr_task.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instr_task" ---
    for thisComponent in instr_task.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instr_task
    instr_task.tStop = globalClock.getTime(format='float')
    instr_task.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instr_task.stopped', instr_task.tStop)
    # check responses
    if quit_instr.keys in ['', [], None]:  # No response was made
        quit_instr.keys = None
    thisExp.addData('quit_instr.keys',quit_instr.keys)
    if quit_instr.keys != None:  # we had a response
        thisExp.addData('quit_instr.rt', quit_instr.rt)
        thisExp.addData('quit_instr.duration', quit_instr.duration)
    thisExp.nextEntry()
    # the Routine "instr_task" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "start_eeg" ---
    # create an object to store info about Routine start_eeg
    start_eeg = data.Routine(
        name='start_eeg',
        components=[EEG_start_text, exit_eeg_text],
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
        
        # *EEG_start_text* updates
        
        # if EEG_start_text is starting this frame...
        if EEG_start_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            EEG_start_text.frameNStart = frameN  # exact frame index
            EEG_start_text.tStart = t  # local t and not account for scr refresh
            EEG_start_text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(EEG_start_text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'EEG_start_text.started')
            # update status
            EEG_start_text.status = STARTED
            EEG_start_text.setAutoDraw(True)
        
        # if EEG_start_text is active this frame...
        if EEG_start_text.status == STARTED:
            # update params
            pass
        
        # if EEG_start_text is stopping this frame...
        if EEG_start_text.status == STARTED:
            # is it time to stop? (based on global clock, using actual start)
            if tThisFlipGlobal > EEG_start_text.tStartRefresh + 5-frameTolerance:
                # keep track of stop time/frame for later
                EEG_start_text.tStop = t  # not accounting for scr refresh
                EEG_start_text.tStopRefresh = tThisFlipGlobal  # on global time
                EEG_start_text.frameNStop = frameN  # exact frame index
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'EEG_start_text.stopped')
                # update status
                EEG_start_text.status = FINISHED
                EEG_start_text.setAutoDraw(False)
        
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
    trials_loop = data.TrialHandler2(
        name='trials_loop',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions(condition_part), 
        seed=None, 
    )
    thisExp.addLoop(trials_loop)  # add the loop to the experiment
    thisTrials_loop = trials_loop.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisTrials_loop.rgb)
    if thisTrials_loop != None:
        for paramName in thisTrials_loop:
            globals()[paramName] = thisTrials_loop[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisTrials_loop in trials_loop:
        currentLoop = trials_loop
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisTrials_loop.rgb)
        if thisTrials_loop != None:
            for paramName in thisTrials_loop:
                globals()[paramName] = thisTrials_loop[paramName]
        
        # --- Prepare to start Routine "image_load" ---
        # create an object to store info about Routine image_load
        image_load = data.Routine(
            name='image_load',
            components=[image_load_text, imag_loader],
        )
        image_load.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for image_load
        image_load.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        image_load.tStart = globalClock.getTime(format='float')
        image_load.status = STARTED
        thisExp.addData('image_load.started', image_load.tStart)
        image_load.maxDuration = None
        # keep track of which components have finished
        image_loadComponents = image_load.components
        for thisComponent in image_load.components:
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
        
        # --- Run Routine "image_load" ---
        # if trial has changed, end Routine now
        if isinstance(trials_loop, data.TrialHandler2) and thisTrials_loop.thisN != trials_loop.thisTrial.thisN:
            continueRoutine = False
        image_load.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.3:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *image_load_text* updates
            
            # if image_load_text is starting this frame...
            if image_load_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                image_load_text.frameNStart = frameN  # exact frame index
                image_load_text.tStart = t  # local t and not account for scr refresh
                image_load_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(image_load_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'image_load_text.started')
                # update status
                image_load_text.status = STARTED
                image_load_text.setAutoDraw(True)
            
            # if image_load_text is active this frame...
            if image_load_text.status == STARTED:
                # update params
                pass
            
            # if image_load_text is stopping this frame...
            if image_load_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > image_load_text.tStartRefresh + 0.3-frameTolerance:
                    # keep track of stop time/frame for later
                    image_load_text.tStop = t  # not accounting for scr refresh
                    image_load_text.tStopRefresh = tThisFlipGlobal  # on global time
                    image_load_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'image_load_text.stopped')
                    # update status
                    image_load_text.status = FINISHED
                    image_load_text.setAutoDraw(False)
            # *imag_loader* period
            
            # if imag_loader is starting this frame...
            if imag_loader.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                imag_loader.frameNStart = frameN  # exact frame index
                imag_loader.tStart = t  # local t and not account for scr refresh
                imag_loader.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(imag_loader, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('imag_loader.started', t)
                # update status
                imag_loader.status = STARTED
                imag_loader.start(0.3)
            elif imag_loader.status == STARTED:  # one frame should pass before updating params and completing
                # Updating other components during *imag_loader*
                cue_stim.setText(cue_text)
                probe_image.setImage(probe_image_file)
                # Component updates done
                imag_loader.complete()  # finish the static period
                imag_loader.tStop = imag_loader.tStart + 0.3  # record stop time
            
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
                image_load.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in image_load.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "image_load" ---
        for thisComponent in image_load.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for image_load
        image_load.tStop = globalClock.getTime(format='float')
        image_load.tStopRefresh = tThisFlipGlobal
        thisExp.addData('image_load.stopped', image_load.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if image_load.maxDurationReached:
            routineTimer.addTime(-image_load.maxDuration)
        elif image_load.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.300000)
        
        # --- Prepare to start Routine "trial" ---
        # create an object to store info about Routine trial
        trial = data.Routine(
            name='trial',
            components=[cue_stim, fixation_stim, probe_image, key_resp_probe, resp_instr],
        )
        trial.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # Run 'Begin Routine' code from trial_code
        
        # This code runs before each trial starts.
        # It calculates the random ISI and sets the start time for the probe.
        isi_duration = random.uniform(1.0, 3.0)
        probe_start_time = 2.0 + 10.0 + isi_duration # cue_dur + sim_dur + isi_dur
        
        response_start_instr = probe_start_time + 2.0
        
        # create starting attributes for key_resp_probe
        key_resp_probe.keys = []
        key_resp_probe.rt = []
        _key_resp_probe_allKeys = []
        # allowedKeys looks like a variable, so make sure it exists locally
        if 'allowed_keys_list' in globals():
            allowed_keys_list = globals()['allowed_keys_list']
        resp_instr.setText('add instr')
        # store start times for trial
        trial.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        trial.tStart = globalClock.getTime(format='float')
        trial.status = STARTED
        thisExp.addData('trial.started', trial.tStart)
        trial.maxDuration = None
        # keep track of which components have finished
        trialComponents = trial.components
        for thisComponent in trial.components:
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
        
        # --- Run Routine "trial" ---
        # if trial has changed, end Routine now
        if isinstance(trials_loop, data.TrialHandler2) and thisTrials_loop.thisN != trials_loop.thisTrial.thisN:
            continueRoutine = False
        trial.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *cue_stim* updates
            
            # if cue_stim is starting this frame...
            if cue_stim.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                cue_stim.frameNStart = frameN  # exact frame index
                cue_stim.tStart = t  # local t and not account for scr refresh
                cue_stim.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(cue_stim, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'cue_stim.started')
                # update status
                cue_stim.status = STARTED
                cue_stim.setAutoDraw(True)
            
            # if cue_stim is active this frame...
            if cue_stim.status == STARTED:
                # update params
                pass
            
            # if cue_stim is stopping this frame...
            if cue_stim.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > cue_stim.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    cue_stim.tStop = t  # not accounting for scr refresh
                    cue_stim.tStopRefresh = tThisFlipGlobal  # on global time
                    cue_stim.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'cue_stim.stopped')
                    # update status
                    cue_stim.status = FINISHED
                    cue_stim.setAutoDraw(False)
            
            # *fixation_stim* updates
            
            # if fixation_stim is starting this frame...
            if fixation_stim.status == NOT_STARTED and tThisFlip >= 2-frameTolerance:
                # keep track of start time/frame for later
                fixation_stim.frameNStart = frameN  # exact frame index
                fixation_stim.tStart = t  # local t and not account for scr refresh
                fixation_stim.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixation_stim, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixation_stim.started')
                # update status
                fixation_stim.status = STARTED
                fixation_stim.setAutoDraw(True)
            
            # if fixation_stim is active this frame...
            if fixation_stim.status == STARTED:
                # update params
                pass
            
            # if fixation_stim is stopping this frame...
            if fixation_stim.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixation_stim.tStartRefresh + 10-frameTolerance:
                    # keep track of stop time/frame for later
                    fixation_stim.tStop = t  # not accounting for scr refresh
                    fixation_stim.tStopRefresh = tThisFlipGlobal  # on global time
                    fixation_stim.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixation_stim.stopped')
                    # update status
                    fixation_stim.status = FINISHED
                    fixation_stim.setAutoDraw(False)
            
            # *probe_image* updates
            
            # if probe_image is starting this frame...
            if probe_image.status == NOT_STARTED and tThisFlip >= probe_start_time-frameTolerance:
                # keep track of start time/frame for later
                probe_image.frameNStart = frameN  # exact frame index
                probe_image.tStart = t  # local t and not account for scr refresh
                probe_image.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(probe_image, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'probe_image.started')
                # update status
                probe_image.status = STARTED
                probe_image.setAutoDraw(True)
            
            # if probe_image is active this frame...
            if probe_image.status == STARTED:
                # update params
                pass
            
            # if probe_image is stopping this frame...
            if probe_image.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > probe_image.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    probe_image.tStop = t  # not accounting for scr refresh
                    probe_image.tStopRefresh = tThisFlipGlobal  # on global time
                    probe_image.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'probe_image.stopped')
                    # update status
                    probe_image.status = FINISHED
                    probe_image.setAutoDraw(False)
            
            # *key_resp_probe* updates
            waitOnFlip = False
            
            # if key_resp_probe is starting this frame...
            if key_resp_probe.status == NOT_STARTED and tThisFlip >= probe_start_time-frameTolerance:
                # keep track of start time/frame for later
                key_resp_probe.frameNStart = frameN  # exact frame index
                key_resp_probe.tStart = t  # local t and not account for scr refresh
                key_resp_probe.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_probe, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_probe.started')
                # update status
                key_resp_probe.status = STARTED
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
                win.callOnFlip(key_resp_probe.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_probe.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_probe.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_probe.getKeys(keyList=list(allowed_keys_list), ignoreKeys=["escape"], waitRelease=False)
                _key_resp_probe_allKeys.extend(theseKeys)
                if len(_key_resp_probe_allKeys):
                    key_resp_probe.keys = _key_resp_probe_allKeys[-1].name  # just the last key pressed
                    key_resp_probe.rt = _key_resp_probe_allKeys[-1].rt
                    key_resp_probe.duration = _key_resp_probe_allKeys[-1].duration
                    # was this correct?
                    if (key_resp_probe.keys == str(correct_response)) or (key_resp_probe.keys == correct_response):
                        key_resp_probe.corr = 1
                    else:
                        key_resp_probe.corr = 0
                    # a response ends the routine
                    continueRoutine = False
            
            # *resp_instr* updates
            
            # if resp_instr is starting this frame...
            if resp_instr.status == NOT_STARTED and tThisFlip >= response_start_instr-frameTolerance:
                # keep track of start time/frame for later
                resp_instr.frameNStart = frameN  # exact frame index
                resp_instr.tStart = t  # local t and not account for scr refresh
                resp_instr.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(resp_instr, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'resp_instr.started')
                # update status
                resp_instr.status = STARTED
                resp_instr.setAutoDraw(True)
            
            # if resp_instr is active this frame...
            if resp_instr.status == STARTED:
                # update params
                pass
            
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
                trial.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in trial.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "trial" ---
        for thisComponent in trial.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for trial
        trial.tStop = globalClock.getTime(format='float')
        trial.tStopRefresh = tThisFlipGlobal
        thisExp.addData('trial.stopped', trial.tStop)
        # check responses
        if key_resp_probe.keys in ['', [], None]:  # No response was made
            key_resp_probe.keys = None
            # was no response the correct answer?!
            if str(correct_response).lower() == 'none':
               key_resp_probe.corr = 1;  # correct non-response
            else:
               key_resp_probe.corr = 0;  # failed to respond (incorrectly)
        # store data for trials_loop (TrialHandler)
        trials_loop.addData('key_resp_probe.keys',key_resp_probe.keys)
        trials_loop.addData('key_resp_probe.corr', key_resp_probe.corr)
        if key_resp_probe.keys != None:  # we had a response
            trials_loop.addData('key_resp_probe.rt', key_resp_probe.rt)
            trials_loop.addData('key_resp_probe.duration', key_resp_probe.duration)
        # the Routine "trial" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "rating" ---
        # create an object to store info about Routine rating
        rating = data.Routine(
            name='rating',
            components=[rating_vividness, text_rating_question, key_resp_rating],
        )
        rating.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        rating_vividness.reset()
        # create starting attributes for key_resp_rating
        key_resp_rating.keys = []
        key_resp_rating.rt = []
        _key_resp_rating_allKeys = []
        # store start times for rating
        rating.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        rating.tStart = globalClock.getTime(format='float')
        rating.status = STARTED
        thisExp.addData('rating.started', rating.tStart)
        rating.maxDuration = None
        # keep track of which components have finished
        ratingComponents = rating.components
        for thisComponent in rating.components:
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
        
        # --- Run Routine "rating" ---
        # if trial has changed, end Routine now
        if isinstance(trials_loop, data.TrialHandler2) and thisTrials_loop.thisN != trials_loop.thisTrial.thisN:
            continueRoutine = False
        rating.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *rating_vividness* updates
            
            # if rating_vividness is starting this frame...
            if rating_vividness.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                rating_vividness.frameNStart = frameN  # exact frame index
                rating_vividness.tStart = t  # local t and not account for scr refresh
                rating_vividness.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(rating_vividness, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'rating_vividness.started')
                # update status
                rating_vividness.status = STARTED
                rating_vividness.setAutoDraw(True)
            
            # if rating_vividness is active this frame...
            if rating_vividness.status == STARTED:
                # update params
                pass
            
            # Check rating_vividness for response to end Routine
            if rating_vividness.getRating() is not None and rating_vividness.status == STARTED:
                continueRoutine = False
            
            # *text_rating_question* updates
            
            # if text_rating_question is starting this frame...
            if text_rating_question.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                text_rating_question.frameNStart = frameN  # exact frame index
                text_rating_question.tStart = t  # local t and not account for scr refresh
                text_rating_question.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(text_rating_question, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'text_rating_question.started')
                # update status
                text_rating_question.status = STARTED
                text_rating_question.setAutoDraw(True)
            
            # if text_rating_question is active this frame...
            if text_rating_question.status == STARTED:
                # update params
                pass
            
            # *key_resp_rating* updates
            waitOnFlip = False
            
            # if key_resp_rating is starting this frame...
            if key_resp_rating.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_rating.frameNStart = frameN  # exact frame index
                key_resp_rating.tStart = t  # local t and not account for scr refresh
                key_resp_rating.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_rating, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_rating.started')
                # update status
                key_resp_rating.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_rating.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_rating.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_rating.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_rating.getKeys(keyList=['1', '2', '3', '4'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_rating_allKeys.extend(theseKeys)
                if len(_key_resp_rating_allKeys):
                    key_resp_rating.keys = _key_resp_rating_allKeys[-1].name  # just the last key pressed
                    key_resp_rating.rt = _key_resp_rating_allKeys[-1].rt
                    key_resp_rating.duration = _key_resp_rating_allKeys[-1].duration
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
                rating.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in rating.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "rating" ---
        for thisComponent in rating.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for rating
        rating.tStop = globalClock.getTime(format='float')
        rating.tStopRefresh = tThisFlipGlobal
        thisExp.addData('rating.stopped', rating.tStop)
        trials_loop.addData('rating_vividness.response', rating_vividness.getRating())
        trials_loop.addData('rating_vividness.rt', rating_vividness.getRT())
        # check responses
        if key_resp_rating.keys in ['', [], None]:  # No response was made
            key_resp_rating.keys = None
        trials_loop.addData('key_resp_rating.keys',key_resp_rating.keys)
        if key_resp_rating.keys != None:  # we had a response
            trials_loop.addData('key_resp_rating.rt', key_resp_rating.rt)
            trials_loop.addData('key_resp_rating.duration', key_resp_rating.duration)
        # the Routine "rating" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        
        # --- Prepare to start Routine "block_break" ---
        # create an object to store info about Routine block_break
        block_break = data.Routine(
            name='block_break',
            components=[block_break_text, key_resp_break],
        )
        block_break.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        block_break_text.setText('')
        # create starting attributes for key_resp_break
        key_resp_break.keys = []
        key_resp_break.rt = []
        _key_resp_break_allKeys = []
        # Run 'Begin Routine' code from code_block_break
        # This code determines if the break screen should be shown.
        # `trials_loop.thisN` is the current trial number (starting from 0).
        # We want a break after trial 31 (thisN=31) and 63 (thisN=63).
        
        # The modulo operator (%) gives the remainder of a division.
        # (thisN + 1) % 32 == 0 checks if we are at the end of a block of 32.
        
        # We also don't want to show it after the very last trial.
        if (trials_loop.thisN + 1) % 32 == 0 and (trials_loop.thisN + 1) < trials_loop.nTotal:
            # It's a break time! Update the text.
            block_num = (trials_loop.thisN + 1) // 32
            block_break_text.text = f'You have completed block {block_num} of 3.\n\nPress space to continue.'
        else:
            # It's not a break time, so skip this entire routine.
            continueRoutine = False
        
        block_break_dur = 60
        # store start times for block_break
        block_break.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        block_break.tStart = globalClock.getTime(format='float')
        block_break.status = STARTED
        thisExp.addData('block_break.started', block_break.tStart)
        block_break.maxDuration = None
        # keep track of which components have finished
        block_breakComponents = block_break.components
        for thisComponent in block_break.components:
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
        
        # --- Run Routine "block_break" ---
        # if trial has changed, end Routine now
        if isinstance(trials_loop, data.TrialHandler2) and thisTrials_loop.thisN != trials_loop.thisTrial.thisN:
            continueRoutine = False
        block_break.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine:
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *block_break_text* updates
            
            # if block_break_text is starting this frame...
            if block_break_text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                block_break_text.frameNStart = frameN  # exact frame index
                block_break_text.tStart = t  # local t and not account for scr refresh
                block_break_text.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(block_break_text, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'block_break_text.started')
                # update status
                block_break_text.status = STARTED
                block_break_text.setAutoDraw(True)
            
            # if block_break_text is active this frame...
            if block_break_text.status == STARTED:
                # update params
                pass
            
            # if block_break_text is stopping this frame...
            if block_break_text.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > block_break_text.tStartRefresh + block_break_dur-frameTolerance:
                    # keep track of stop time/frame for later
                    block_break_text.tStop = t  # not accounting for scr refresh
                    block_break_text.tStopRefresh = tThisFlipGlobal  # on global time
                    block_break_text.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'block_break_text.stopped')
                    # update status
                    block_break_text.status = FINISHED
                    block_break_text.setAutoDraw(False)
            
            # *key_resp_break* updates
            waitOnFlip = False
            
            # if key_resp_break is starting this frame...
            if key_resp_break.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                key_resp_break.frameNStart = frameN  # exact frame index
                key_resp_break.tStart = t  # local t and not account for scr refresh
                key_resp_break.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(key_resp_break, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'key_resp_break.started')
                # update status
                key_resp_break.status = STARTED
                # keyboard checking is just starting
                waitOnFlip = True
                win.callOnFlip(key_resp_break.clock.reset)  # t=0 on next screen flip
                win.callOnFlip(key_resp_break.clearEvents, eventType='keyboard')  # clear events on next screen flip
            if key_resp_break.status == STARTED and not waitOnFlip:
                theseKeys = key_resp_break.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
                _key_resp_break_allKeys.extend(theseKeys)
                if len(_key_resp_break_allKeys):
                    key_resp_break.keys = _key_resp_break_allKeys[-1].name  # just the last key pressed
                    key_resp_break.rt = _key_resp_break_allKeys[-1].rt
                    key_resp_break.duration = _key_resp_break_allKeys[-1].duration
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
                block_break.forceEnded = routineForceEnded = True
                break
            continueRoutine = False  # will revert to True if at least one component still running
            for thisComponent in block_break.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "block_break" ---
        for thisComponent in block_break.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for block_break
        block_break.tStop = globalClock.getTime(format='float')
        block_break.tStopRefresh = tThisFlipGlobal
        thisExp.addData('block_break.stopped', block_break.tStop)
        # check responses
        if key_resp_break.keys in ['', [], None]:  # No response was made
            key_resp_break.keys = None
        trials_loop.addData('key_resp_break.keys',key_resp_break.keys)
        if key_resp_break.keys != None:  # we had a response
            trials_loop.addData('key_resp_break.rt', key_resp_break.rt)
            trials_loop.addData('key_resp_break.duration', key_resp_break.duration)
        # the Routine "block_break" was not non-slip safe, so reset the non-slip timer
        routineTimer.reset()
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'trials_loop'
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    # get names of stimulus parameters
    if trials_loop.trialList in ([], [None], None):
        params = []
    else:
        params = trials_loop.trialList[0].keys()
    # save data for this loop
    trials_loop.saveAsText(filename + 'trials_loop.csv', delim=',',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    
    # --- Prepare to start Routine "end_task" ---
    # create an object to store info about Routine end_task
    end_task = data.Routine(
        name='end_task',
        components=[end_text, exit_end],
    )
    end_task.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for exit_end
    exit_end.keys = []
    exit_end.rt = []
    _exit_end_allKeys = []
    # store start times for end_task
    end_task.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    end_task.tStart = globalClock.getTime(format='float')
    end_task.status = STARTED
    thisExp.addData('end_task.started', end_task.tStart)
    end_task.maxDuration = None
    # keep track of which components have finished
    end_taskComponents = end_task.components
    for thisComponent in end_task.components:
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
    
    # --- Run Routine "end_task" ---
    end_task.forceEnded = routineForceEnded = not continueRoutine
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
        
        # *exit_end* updates
        waitOnFlip = False
        
        # if exit_end is starting this frame...
        if exit_end.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            exit_end.frameNStart = frameN  # exact frame index
            exit_end.tStart = t  # local t and not account for scr refresh
            exit_end.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(exit_end, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'exit_end.started')
            # update status
            exit_end.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(exit_end.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(exit_end.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if exit_end.status == STARTED and not waitOnFlip:
            theseKeys = exit_end.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _exit_end_allKeys.extend(theseKeys)
            if len(_exit_end_allKeys):
                exit_end.keys = _exit_end_allKeys[-1].name  # just the last key pressed
                exit_end.rt = _exit_end_allKeys[-1].rt
                exit_end.duration = _exit_end_allKeys[-1].duration
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
            end_task.forceEnded = routineForceEnded = True
            break
        continueRoutine = False  # will revert to True if at least one component still running
        for thisComponent in end_task.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "end_task" ---
    for thisComponent in end_task.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for end_task
    end_task.tStop = globalClock.getTime(format='float')
    end_task.tStopRefresh = tThisFlipGlobal
    thisExp.addData('end_task.stopped', end_task.tStop)
    # check responses
    if exit_end.keys in ['', [], None]:  # No response was made
        exit_end.keys = None
    thisExp.addData('exit_end.keys',exit_end.keys)
    if exit_end.keys != None:  # we had a response
        thisExp.addData('exit_end.rt', exit_end.rt)
        thisExp.addData('exit_end.duration', exit_end.duration)
    thisExp.nextEntry()
    # the Routine "end_task" was not non-slip safe, so reset the non-slip timer
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
