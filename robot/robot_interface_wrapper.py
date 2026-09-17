# FANUC robot interface for Python using Pyton.NET
# pip install pythonnet   # fuer integration mit .NET  # funktioniert nicht mit Python 3.9
# In March 2022, the stable branch is not yet compatible with Python 3.10, so the prerelease needs to be used # pip install --pre pythonnet


# ============#
# == TO-DO: ==#
# ============#
# - System Arrays in die Klasse - DONE
# - Funktion um automatisch benötigtes System Array zu erstellen def make_array(length, type=Int/Double) - DONE
# - amount wird nicht addiert - DONE
# - richtigen Ort für DLL definieren und auswählen, am besten in robot_interface Ordner, sonst spam mit log-files in Hauptorder - DONE, aber aufpassen, directory muss vllt angepasst werden
# - herausfinden ob bei get_cartesian_pos uframe und utool Auswirkungen haben bzw. tatsächlich einstellbar sind; möglicherweise sind das auch nur Werte zum auslesen
# - vllt sinnvoll bei get_cartesian_pos und get_joint_pos die Arrays wie xyzwpr und joints, etc. direkt in Klasse init zu packen -> sauberer Code
# - möglicherweise zusätzliche Funktionen, um aus get_current_pos auch noch config, etc. auszulesen
# - Funktionen für einfache Group IOs hinzufügen, wären identisch zu AO/AI nur index ohne +1000
# ============#
# ============#

import numpy as np
import os, sys, time


class ROB_INF:
    def __init__(self, robot_ip):
        self.robot_ip = robot_ip
        self.dll_dir = "./VisionNext2OEM/"
        self.dllname = "RobotInterfaceDotNet"
        self.path = r"%s%s" % (self.dll_dir, self.dllname)

        print(f"ROB_INF: path: {self.path}")
        print(f"ROB_INF: getcwd: {os.getcwd()}")

        sys.path.append(os.getcwd())
        clr.AddReference(self.path)

        print(f"ROB_INF: connecting to {robot_ip}")

        # Fanuc Robotics Japan Interface
        # import inside class because apparently dll has to be referenced before (maybe not necesary, dll magic
        import FRRJIf
        import System

        self.System = System

        self.system = System.Text
        self.encode = System.Text.Encoding.GetEncoding("UTF-8")

        self.robot_interface = FRRJIf.Core(self.encode)
        self.data_table = self.robot_interface.get_DataTable()

        # robotif Handbuch S.12: AddCurPosUF(FRRJIf.FRIF_DATA_TYPE DataType, int Group, int UF) # Userframe vorgeben!
        self.current_position = self.data_table.AddCurPosUF(
            FRRJIf.FRIF_DATA_TYPE.CURPOS, 1, 1
        )

        # robotif Handbuch S.12: AddPosRegXyzwpr (FRRJIf.FRIF_DATA_TYPE DataType, int Group, int StartIndex, int EndIndex) #by SL
        self.pos_regXyzwpr = self.data_table.AddPosRegXyzwpr(
            FRRJIf.FRIF_DATA_TYPE.POSREG_XYZWPR, 1, 1, 300
        )

        # robotif Handbuch S.14: AddSysVar(DataType As FRIF_DATA_TYPE, SysVarName As String) As DataSysVar
        self.fast_clock = self.data_table.AddSysVar(
            FRRJIf.FRIF_DATA_TYPE.SYSVAR_INT, "$FAST_CLOCK"
        )

        # add space all 1024 flags to datatable ?
        self.flag_table = self.data_table.AddFlag(1, 1024)

        # 5.3.1.7 AddSysVar - Register system variables (integer, real and string type) to DataTable
        # self.sysvar_force = self.data_table.AddSysVar(FRRJIf.FRIF_DATA_TYPE.SYSVAR_ARRAY[FRRJIf.FRIF_DATA_TYPE.SYSVAR_REAL], "$CCC_GRP[1].$FS_FORCE") # AttributeError: type object 'FRIF_DATA_TYPE' has no attribute 'SYSVAR_ARRAY'
        self.sysvar_force = []
        for i in range(0, 6):
            self.sysvar_force.append(
                self.data_table.AddSysVar(
                    FRRJIf.FRIF_DATA_TYPE.SYSVAR_REAL, f"$CCC_GRP[1].$FS_FORCE[{i+1}]"
                )
            )
        # GEHT AUCH DIREKT EIN ARRAY? VIELLEICHT SCHNELLER?
        # = self.System.Array[self.System.Double]([0, 0, 0, 0, 0, 0, 0])

    # connect RobotInterface to robot; has to be called right after class initialization
    def connect(self):
        # Provide robot IP address for connection
        self.robot_interface.set_TimeOutValue(5)
        if not self.robot_interface.Connect(self.robot_ip):
            print("Cannot connect RobotInterface")
            sys.exit("no connection")
        else:
            print("RobotInterface connected")

    ### not clear, was das eigentlich ausgibt
    def get_clock(self):
        fclock = self.System.Int32(0)
        self.data_table.Refresh()
        # System variable $FAST_CLOCK
        return_value = self.fast_clock.GetValue(fclock, self.encode)
        print(self.fast_clock.SysVarName(), ": ", return_value[1])
        return return_value

    def get_force(self):
        force = self.System.Double(0)
        self.data_table.Refresh()
        # System variable $FAST_CLOCK
        force_arr = []
        for sysvar_force_single in self.sysvar_force:

            return_value = sysvar_force_single.GetValue(force, self.encode)
            # force_arr.append([row for row in return_value])
            force_arr.append(return_value[1])

        # print (self.sysvar_force[0].SysVarName(), ': ', force_arr)
        return force_arr

    # Returns current cartesian robot position
    # since it is the .GetValue method it's unclear, if setting uframe and utool actually does something
    def get_cartesian_pos(self, user_frame_number, user_tool_number):

        # Xyzwpr() should have 9 elements (robot 6 axes + 3 extended axes)
        xyzwpr = self.System.Array[self.System.Double]([0, 0, 0, 0, 0, 0, 0, 0, 0])

        # Arugument Config() will have returned configuration of current position
        # Unclear what it does? Maybe it should be variable in params?
        config = self.System.Array[self.System.Double]([0, 0, 0, 0, 0, 0, 0])

        # Argument Joint() will have returned joint values. Joint() should have 9 elements (robot 6 axes + 3 extended axes)
        joint = self.System.Array[self.System.Double]([0, 0, 0, 0, 0, 0, 0, 0, 0])

        # Argument UF will have returned user frame number
        uframe_num = self.System.Int16(user_frame_number)

        # Argument UT will have returned user tool number
        utool_num = self.System.Int16(user_tool_number)

        # When current position has valid Cartesian values, argument ValidC will have non 0
        # ??? not sure what that does
        valid_c = self.System.Int16(0)

        # When current position do not have valid joint values, argument ValidJ will have 0
        ### also not sure what that one does
        valid_j = self.System.Int16(0)

        # ALWAYS use data_table.Refresh() to update data, otherwise error
        self.data_table.Refresh()

        return_value = self.current_position.GetValue(
            xyzwpr, config, joint, uframe_num, utool_num, valid_c, valid_j
        )
        # 5.4.1.2 GetValue - Read current position      ref System.Array Xyzwpr, ref System.Array Config, ref System.Array Joint, ref short UF, ref short UT, ref short ValidC, ref short ValidJ
        print("return value", return_value)

        pos = [row for row in return_value[1]]
        con = [row for row in return_value[2]]
        print(
            "Current Cartesian Position with UFRAME ",
            uframe_num,
            " and UTOOL ",
            utool_num,
            ":  ",
            pos,
            con,
        )
        return pos

    # returns current joint positions
    def get_joint_pos(self):
        # Xyzwpr() should have 9 elements (robot 6 axes + 3 extended axes)
        xyzwpr = self.System.Array[self.System.Double]([0, 0, 0, 0, 0, 0, 0, 0, 0])

        # Arugument Config() will have returned configuration of current position
        # Unclear what it does? Maybe it should be variable in params?
        config = self.System.Array[self.System.Double]([0, 0, 0, 0, 0, 0, 0])

        # Argument Joint() will have returned joint values. Joint() should have 9 elements (robot 6 axes + 3 extended axes)
        joint = self.System.Array[self.System.Double]([0, 0, 0, 0, 0, 0, 0, 0, 0])

        # Argument UF will have returned user frame number
        uframe_num = self.System.Int16(0)

        # Argument UT will have returned user tool number
        utool_num = self.System.Int16(1)

        # When current position has valid Cartesian values, argument ValidC will have non 0
        valid_c = self.System.Int16(0)

        # When current position do not have valid joint values, argument ValidJ will have 0
        valid_j = self.System.Int16(0)

        # ALWAYS use data_table.Refresh() to update data, otherwise error
        self.data_table.Refresh()

        return_value = self.current_position.GetValue(
            xyzwpr, config, joint, uframe_num, utool_num, valid_c, valid_j
        )
        joint_pos = [row for row in return_value[3]]
        print("Current Joint Position:   ", joint_pos)
        return joint_pos

    def get_pos_reg(self, user_frame_number, user_tool_number, index_number):
        print(f"ROB_INF get_pos_reg")
        # bool GetValue(int Index, ref System.Array Xyzwpr, ref System.Array Config, ref System.Array Joint, ref short UF, ref short UT, ref short ValidC, ref short ValidJ)

        # int Index
        index = self.System.Int32(index_number)

        # Argument System.Array Xyzwpr() should have 9 elements (robot 6 axes + 3 extended axes)
        xyzwpr = self.System.Array[self.System.Double]([0, 0, 0, 0, 0, 0, 0, 0, 0])

        # Argument System.Array Config() will have returned configuration of current position
        config = self.System.Array[self.System.Double]([0, 0, 0, 0, 0, 0, 0])

        # Argument System.Array Joint() will have returned joint values. Joint() should have 9 elements (robot 6 axes + 3 extended axes)
        joint = self.System.Array[self.System.Double]([0, 0, 0, 0, 0, 0, 0, 0, 0])

        # Argument short UF will have returned user frame number
        uframe_num = self.System.Int16(user_frame_number)

        # Argument short UT will have returned user tool number
        utool_num = self.System.Int16(user_tool_number)

        # When current position has valid Cartesian values, argument ValidC will have non 0
        # ??? not sure what that does
        valid_c = self.System.Int16(0)

        # When current position do not have valid joint values, argument ValidJ will have 0
        ### also not sure what that one does
        valid_j = self.System.Int16(0)

        # ALWAYS use data_table.Refresh() to update data, otherwise error
        self.data_table.Refresh()

        return_value = self.pos_reg.GetValue(
            index, xyzwpr, config, joint, uframe_num, utool_num, valid_c, valid_j
        )
        # 5.6.1.2 GetValue - position register value

        """
        pos_c = [row for row in return_value[1]]
        con_c = [row for row in return_value[2]]
        pos_j = [row for row in return_value[3]]
        """
        """
        print("ROB_INF return_value")
        print(return_value)
        print(return_value[0]) # bool status
        print(pos_c) # system.array Xyzwpr
        print(con_c) # system.array Config
        print(pos_j) # system.array Joint
        print(return_value[4]) # short UF
        print(return_value[5]) # short UT
        print(return_value[6]) # short Valid_C
        print(return_value[7]) # short Valid_J
        
        print('Cartesian Position from Reg ', index, ' with UFRAME ', uframe_num, ' and UTOOL ', utool_num, ':  ', pos_c)
        print('Config from Reg ', index, ' with UFRAME ', uframe_num, ' and UTOOL ', utool_num, ':  ', con_c)
        print('Joint Position from Reg ', index, ' with UFRAME ', uframe_num, ' and UTOOL ', utool_num, ':  ', pos_j)
        """
        return return_value

    def set_pos_reg(self, index, xyzwpr, config):

        # Argument System.Array Xyzwpr() should have 9 elements (robot 6 axes + 3 extended axes)
        xyzwpr_System = self.System.Array[self.System.Single](xyzwpr)

        # Argument System.Array Config()
        config_System = self.System.Array[self.System.Int16](config)

        return_value, ar1, ar2 = self.pos_regXyzwpr.SetValueXyzwpr(
            index, xyzwpr_System, config_System
        )
        
        print("")
        print(index)
        print(xyzwpr)
        print(config)
        print("")
        
        if return_value == None:
            print("Error")
            print("Fehler beim setzen des Positionsregisters, None")
            print("Error")

        if return_value == False:
            print("Error")
            print("Fehler beim setzen des Positionsregisters , False")
            print("Error")
        # ALWAYS use data_table.Refresh() to update data, otherwise error
        self.data_table.Refresh()
        self.pos_regXyzwpr.Update()

        print(return_value)

    #####################
    #####################
    ## ANALOGE DO/DI ####
    #####################

    ### ''''''''''''''''''###
    ## Int32 verwenden!!!! ##

    # Reads, prints and returns Analog Outputs
    # PARAMS: - index_start = index of first AO you want to read
    #         - amount = how many AOs you want to read following and including AO[index_start]
    # Group input 1. Logic is always (start, array, count)
    # funktioniert: lesen von groups und analogen inputs und outputs(index + 1000)
    def get_AO(self, index_start: int, amount: int):
        array_to_fill = self.System.Array[self.System.Int32](
            np.zeros(amount), dtype=self.System.Int16
        )

        # array_to_fill = self.array_to_fill
        index_start = index_start + 1000
        return_value = self.robot_interface.ReadGO(index_start, array_to_fill, amount)
        current_values = [row for row in return_value[1]]
        # print (f"AO{index_start-1000} to AO{(index_start-1000)+amount-1}", current_values)
        return current_values

    def set_AO(self, index_start, array_new_values):
        """write Analog Output
        sets Analog Outputs to the values in array_new_values starting from the index_start AO
        PARAMETER: 1: start index, 2: System.Array mit den zu setzenden Werten, 3: Anzahl der zu setzenden Werte, also <= len(array)
        Achtung: wenn man die AOs mit dem TeachPendant nicht setzen kann (z.B. Fehler "Port is not assiged"), dann funktioniert die
        Methode write_AO nicht. Es wird keine Fehlermeldung angezeigt"""

        index_start = index_start + 1000
        array_new_values_converted = self.System.Array[self.System.Int32](
            array_new_values
        )
        self.robot_interface.WriteGO(
            index_start, array_new_values_converted, len(array_new_values)
        )
        print(
            f"RI - Set values from AO{index_start} to AO{index_start + len(array_new_values) - 1} to: {array_new_values}"
        )

    def get_AI(self, index_start, amount):
        """reads, prints and returns Analog Inputs: funktioniert genauso wie bei AOs
        PARAMS: - index_start = index of first AI you want to read
                - amount = how many AIs you want to read following and including AI[index_start]
        Group input 1. Logic is always (start, array, count)
        funktioniert lesen von groups und analogen inputs und outputs(index + 1000)"""
        array_to_fill = self.System.Array[self.System.Int32](
            np.zeros(amount, dtype=self.System.Int32)
        )
        index_start = index_start + 1000
        return_value = self.robot_interface.ReadGI(index_start, array_to_fill, amount)
        current_values = [row for row in return_value[1]][0]
        print(f"RI - Get values from AO{index_start}: {current_values}")
        return current_values

    #### END ANALOG AOs ####

    ####################
    ####################
    ## DIGITALE DO/DI ##
    ####################

    ### ''''''''''''''''''###
    ### auf Int16 lassen! ###

    # reads, prints and returns Digital Outputs
    # PARAMS: - index_start = index of first DO you want to read
    #         - amount = how many DOs you want to read following and including DO[index_start]
    # fuer weitere Entwicklung: SDO sind die normalen Digital Outputs: bei den anderen treten folgende Fehler auf:
    #                           - FIXED! UPDATE: immer nur in Abschnitten von maximal 20 Werten auslesbar, erster Parameter gibt Startindex an
    #                           - FIXED! UPDATE 2: anscheinend nur die ersten 20 DOs aufrufbar
    def get_DO(self, index_start, amount):
        array_to_fill = self.System.Array[self.System.Int16](
            np.zeros(amount), dtype=self.System.Int16
        )
        return_value = self.robot_interface.ReadSDO(index_start, array_to_fill, amount)
        current_values = [row for row in return_value[1]]
        # print (f"DO{index_start} to DO{index_start + amount - 1}:", current_values)
        return current_values

    # write Digital Output
    # sets Digital Outputs to the values in array_new_values starting from the index_start AO
    # array_new_values: Integer Array! either 0 or 1!!!
    def set_DO(self, index_start, array_new_values):
        array_new_values_converted = self.System.Array[self.System.Int16](
            array_new_values
        )
        self.robot_interface.WriteSDO(
            index_start, array_new_values_converted, len(array_new_values)
        )
        print(
            f"Set values from DO{index_start} to DO{index_start + len(array_new_values) - 1} to: {array_new_values}"
        )

    # reads, prints and returns Digital Input
    # PARAMS: - index_start = index of first DI you want to read
    #         - amount = how many DIs you want to read following and including DI[index_start]
    def get_DI(self, index_start, amount: int):
        array_to_fill = self.System.Array[self.System.Int16](
            np.zeros(amount, dtype=self.System.Int16)
        )
        return_value = self.robot_interface.ReadSDI(index_start, array_to_fill, amount)
        current_values = [row for row in return_value[1]]
        # print (f"DI{index_start} to DI{index_start + amount - 1}", current_values)
        return current_values

    #### END DIGITAL IOs ####
    #########################

    ####################
    ####################
    ### ROBOT DO/DI ###
    ####################

    ### ''''''''''''''''''###
    ### auf Int16 lassen! ###

    # reads, prints and returns Robot Outputs
    # PARAMS: - index_start = index of first DO you want to read
    #         - amount = how many ROs you want to read following and including RO[index_start]
    def get_RO(self, index_start, amount):
        array_to_fill = self.System.Array[self.System.Int16](
            np.zeros(amount), dtype=self.System.Int16
        )
        return_value = self.robot_interface.ReadRDO(index_start, array_to_fill, amount)
        current_values = [row for row in return_value[1]]
        print(f"RO{index_start} to RO{index_start + amount - 1}:", current_values)
        return current_values

    # write Robot Output
    # sets Robot Outputs to the values in array_new_values starting from the index_start AO
    # array_new_values: Integer Array! either 0 or 1!!!
    def set_RO(self, index_start, array_new_values):
        array_new_values_converted = self.System.Array[self.System.Int16](
            array_new_values
        )
        self.robot_interface.WriteRDO(
            index_start, array_new_values_converted, len(array_new_values)
        )
        print(
            f"Set values from RO{index_start} to RO{index_start + len(array_new_values) - 1} to: {array_new_values}"
        )

    # reads, prints and returns Robot Input
    # PARAMS: - index_start = index of first DI you want to read
    #         - amount = how many DIs you want to read following and including DI[index_start]
    def get_RI(self, index_start, amount):
        array_to_fill = self.System.Array[self.System.Int16](
            np.zeros(amount), dtype=self.System.Int16
        )
        return_value = self.robot_interface.ReadRDI(index_start, array_to_fill, amount)
        current_values = [row for row in return_value[1]]
        print(f"RI{index_start} to RI{index_start + amount - 1}", current_values)
        return current_values

    #### END Robot IOs ####
    #########################

    ##################
    ##### FLAGS ######
    ##################
    # Reads and returns value of specific Flag
    # ERSTER PARAMETER BESCHREIBT DEN INDEX DES FLAG
    # Second parameter describes what to return if error
    # Returns tuple with: 1. Erfolgreiche Abfrage -> True/False
    #                     2. Flag ON/OFF -> 1/0
    def get_flag(self, index, error_return_value):
        # ALWAYS use data_table.Refresh() to update data, otherwise error
        self.data_table.Refresh()
        return_value = self.flag_table.GetValue(index, error_return_value)
        #print(f"FLAG:{index}", return_value)
        return return_value[1]

    # Write Flags
    # ERSTER PARAMETER BESCHREIBT DEN INDEX DES FLAG
    # Second Parameter: array with new values for Flags from index_start to (index_start + len(array)); values must be 1 or 0
    def set_flag(self, index_start, array_new_values):
        array_new_values_converted = self.System.Array[self.System.Int16](
            array_new_values
        )
        self.flag_table.SetValues(
            index_start, array_new_values_converted, len(array_new_values)
        )
        print(
            f"Set values from Flag {index_start} to Flag {index_start + len(array_new_values) - 1} to: {array_new_values}"
        )

    #### END FLAGS ####
    ###################


class ROB_INF_test:
    def __init__(self, robot_ip):
        self.robot_ip = robot_ip
        self.dll_dir = "./robot_interface/"
        self.dllname = "RobotInterfaceDotNet"
        self.path = r"%s%s" % (self.dll_dir, self.dllname)
        sys.path.append(os.getcwd())
        clr.AddReference(self.path)

        # Fanuc Robotics Japan Interface
        # import inside class because apparently dll has to be referenced before (maybe not necesary, dll magic
        import FRRJIf
        import System

        self.System = System

    # connect RobotInterface to robot; has to be called right after class initialization
    def connect(self):
        # Provide robot IP address for connection
        pass

    ### not clear, was das eigentlich ausgibt
    def get_clock(self):
        return_value = None
        return return_value

    # Returns current cartesian robot position
    # since it is the .GetValue method it's unclear, if setting uframe and utool actually does something
    def get_cartesian_pos(self, user_frame_number, user_tool_number):
        pos = None
        return pos

    # returns current joint positions
    def get_joint_pos(self):
        joint_pos = None
        return joint_pos

    def get_pos_reg(self, user_frame_number, user_tool_number, index_number):
        # bool GetValue(int Index, ref System.Array Xyzwpr, ref System.Array Config, ref System.Array Joint, ref short UF, ref short UT, ref short ValidC, ref short ValidJ)

        # int Index
        index = self.System.Int32(index_number)

        # Argument System.Array Xyzwpr() should have 9 elements (robot 6 axes + 3 extended axes)
        xyzwpr = self.System.Array[self.System.Double]([0, 0, 0, 0, 0, 0, 0, 0, 0])

        # Argument System.Array Config() will have returned configuration of current position
        config = self.System.Array[self.System.Double]([0, 0, 0, 0, 0, 0, 0])

        # Argument System.Array Joint() will have returned joint values. Joint() should have 9 elements (robot 6 axes + 3 extended axes)
        joint = self.System.Array[self.System.Double]([0, 0, 0, 0, 0, 0, 0, 0, 0])

        # Argument short UF will have returned user frame number
        uframe_num = self.System.Int16(user_frame_number)

        # Argument short UT will have returned user tool number
        utool_num = self.System.Int16(user_tool_number)

        # When current position has valid Cartesian values, argument ValidC will have non 0
        # ??? not sure what that does
        valid_c = self.System.Int16(0)

        # When current position do not have valid joint values, argument ValidJ will have 0
        ### also not sure what that one does
        valid_j = self.System.Int16(0)

        pos = [row for row in xyzwpr]

        print(
            "Cartesian Position from Reg ",
            index,
            " with UFRAME ",
            uframe_num,
            " and UTOOL ",
            utool_num,
            ":  ",
            pos,
        )
        return pos

    def set_pos_reg(
        self,
        user_frame_number,
        user_tool_number,
        index_number,
        xyzwpr_array9,
        config_array6,
    ):

        # int Index
        index = self.System.Int32(index_number)

        # Argument System.Array Xyzwpr() should have 9 elements (robot 6 axes + 3 extended axes)
        xyzwpr = self.System.Array[self.System.Double](xyzwpr_array9)

        # Argument System.Array Config() will have returned configuration of current position
        config = self.System.Array[self.System.Double](config_array6)

        # Argument short UF will have returned user frame number
        uframe_num = self.System.Int16(user_frame_number)

        # Argument short UT will have returned user tool number
        utool_num = self.System.Int16(user_tool_number)

        pos = [row for row in xyzwpr]

        print(
            "Set cartesian Position to Reg ",
            index,
            " with UFRAME ",
            uframe_num,
            " and UTOOL ",
            utool_num,
            ":  ",
            pos,
        )

    #####################
    #####################
    ## ANALOGE DO/DI ####
    #####################

    ### ''''''''''''''''''###
    ## Int32 verwenden!!!! ##

    # Reads, prints and returns Analog Outputs
    # PARAMS: - index_start = index of first AO you want to read
    #         - amount = how many AOs you want to read following and including AO[index_start]
    # Group input 1. Logic is always (start, array, count)
    # funktioniert: lesen von groups und analogen inputs und outputs(index + 1000)
    def get_AO(self, index_start: int, amount: int):
        current_values = 6 * [0]
        return current_values

    def set_AO(self, index_start, array_new_values):
        pass

    def get_AI(self, index_start, amount):
        current_values = 100 * [0]
        return current_values[0]

    def get_DO(self, index_start, amount):
        current_values = 20 * [0]
        return current_values

    def set_DO(self, index_start, array_new_values):
        pass

    def get_DI(self, index_start, amount: int):
        current_values = 20 * [0]
        return current_values

    def get_RO(self, index_start, amount):
        current_values = 20 * [0]
        return current_values

    def set_RO(self, index_start, array_new_values):
        print(
            f"Set values from RO{index_start} to RO{index_start + len(array_new_values) - 1} to: {array_new_values}"
        )

    # reads, prints and returns Robot Input
    # PARAMS: - index_start = index of first DI you want to read
    #         - amount = how many DIs you want to read following and including DI[index_start]
    def get_RI(self, index_start, amount):
        current_values = 20 * [0]
        print(f"RI{index_start} to RI{index_start + amount - 1}", current_values)
        return current_values

    #### END Robot IOs ####
    #########################

    ##################
    ##### FLAGS ######
    ##################
    # Reads and returns value of specific Flag
    # ERSTER PARAMETER BESCHREIBT DEN INDEX DES FLAG
    # Second parameter describes what to return if error
    # Returns tuple with: 1. Erfolgreiche Abfrage -> True/False
    #                     2. Flag ON/OFF -> 1/0
    def get_flag(self, index, error_return_value):
        # ALWAYS use data_table.Refresh() to update data, otherwise error
        return_value = 20 * [0]
        print(f"FLAG:{index}", return_value)
        return return_value

    # Write Flags
    # ERSTER PARAMETER BESCHREIBT DEN INDEX DES FLAG
    # Second Parameter: array with new values for Flags from index_start to (index_start + len(array)); values must be 1 or 0
    def set_flag(self, index_start, array_new_values):
        print(
            f"Set values from Flag {index_start} to Flag {index_start + len(array_new_values) - 1} to: {array_new_values}"
        )

    #### END FLAGS ####
    ###################
