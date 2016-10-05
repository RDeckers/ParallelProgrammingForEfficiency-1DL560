rwildcard=$(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2) $(filter $(subst *, %,$2),$d))
directories=$(sort $(dir $(wildcard $/*)))
ifeq ($(OS),Windows_NT)
	FIX_PATH = $(subst /,\,$1)
	REPORT = @echo $1
	CHK_DIR_EXISTS = if not exist "$(strip $1)" mkdir "$(strip $1)"
	NUKE = rmdir /s /q
	COPY_DIR = xcopy $(call FIX_PATH,$1 $2) /E /H /Y
	COPY_CONTENT = xcopy /s /Y $(call FIX_PATH,$1 $2)
	COPY = xcopy $(call FIX_PATH,$1 $2) /Y
	INSTALL_LIB_DIR := Z:/lib/
	INSTALL_BIN_DIR := Z:/bin/
	INSTALL_INCLUDE_DIR := Z:/include/
	LIB_SUFFIX =.dll
	OPENCL_HEADERS :=$(call rwildcard, Z:/include/CL/, *.h)
	BINARY_EXTENSION := .exe
else
	REPORT = @echo "\033[4;1;37m$1\033[0m"
	CHK_DIR_EXISTS = test -d $1 || mkdir -p $1
	NUKE = rm -r $1
	COPY_DIR = cp -rv $1 $2
	FIX_PATH = $1
	INSTALL_LIB_DIR := ~/lib/
	INSTALL_BIN_DIR := ~/bin/
	INSTALL_INCLUDE_DIR := ~/include/
	LIB_SUFFIX :=.so
	OPENCL_HEADERS :=/usr/include/CL/*.h
	BINARY_EXTENSION :=
endif

PROJECT_DIR :=$(dir $(realpath $(lastword $(MAKEFILE_LIST))))
OBJ_DIR := $(PROJECT_DIR)obj
BIN_DIR := $(PROJECT_DIR)bin
SRC_DIR := $(PROJECT_DIR)src
SCRIPT_DIR := $(PROJECT_DIR)scripts
PLOTS_DIR := $(PROJECT_DIR)plots
DATA_DIR := $(PROJECT_DIR)data

PLOTS_GNUPLOT := $(call rwildcard, $(PLOTS_DIR), *.gnuplot)
PLOTS_PNG := $(patsubst %.gnuplot, %.png, $(PLOTS_GNUPLOT))
CPP_FILES := $(wildcard $(SRC_DIR)/*.cpp)
SCRIPTS := $(wildcard $(SCRIPT_DIR)/*.sh)
BINARIES := $(patsubst $(SRC_DIR)/%.cpp,$(BIN_DIR)/%$(BINARY_EXTENSION),$(CPP_FILES))

COMMON_CPP_FILES := $(wildcard $(SRC_DIR)/common/*.cpp)
OBJ_FILES = $(patsubst $(SRC_DIR)/common/%.cpp,$(OBJ_DIR)/%.o,$(COMMON_CPP_FILES))
DATA_FILES := $(patsubst $(PLOTS_DIR)/%.gnuplot,$(DATA_DIR)/%.dat,$(PLOTS_GNUPLOT))

LD_FLAGS += --std=c++11 -march=native -lOpenCL -lclu -lutilities -lm -ltiff -lxml2 -L $(INSTALL_LIB_DIR)
CPP_FLAGS += --std=c++11 -g -O2 -pipe -march=native -I $(PROJECT_DIR)headers -I $(INSTALL_INCLUDE_DIR) -I /usr/include/libxml2 -I Z:/msys64/mingw64/include/libxml2/

.SECONDARY: $(OBJ_FILES) $(DATA_FILES)

binaries: $(BINARIES)

report: plots
	$(call REPORT, Making PDF)
	pdflatex -interaction=nonstopmode $(PROJECT_DIR)main.tex

plots: $(PLOTS_PNG)

data: $(DATA_FILES)

$(DATA_DIR)/%.dat : $(BIN_DIR)/%$(BINARY_EXTENSION)
	$(call REPORT,Running $<)
	$(call CHK_DIR_EXISTS, $(DATA_DIR))
	$< > $@

$(PLOTS_DIR)/%.png : $(DATA_FILES) $(PLOTS_DIR)/%.gnuplot
	$(call REPORT,Building $@)
	gnuplot -e "output_file='$@';term_type='png'" -c "$(patsubst %.png,%.gnuplot,$@)"

$(BIN_DIR)%$(BINARY_EXTENSION) : $(SRC_DIR)/%.cpp $(OBJ_FILES)
	$(call REPORT,Building $@)
	$(call CHK_DIR_EXISTS, $(dir $@))
	g++ $(CPP_FLAGS) -o "$@" "$<" $(OBJ_FILES) $(LD_FLAGS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/common/%.cpp
	$(call REPORT,Compiling $@)
	$(call CHK_DIR_EXISTS, $(dir $@))
	g++ $(CPP_FLAGS) -o "$@" -c "$<"

clean:
	$(call REPORT,Cleaning...)
	-$(NUKE) "$(OBJ_DIR)" "$(BIN_DIR)"
