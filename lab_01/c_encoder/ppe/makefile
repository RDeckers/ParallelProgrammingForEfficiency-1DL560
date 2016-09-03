rwildcard=$(foreach d,$(wildcard $1*),$(call rwildcard,$d/,$2) $(filter $(subst *, %,$2),$d))
ifeq ($(OS),Windows_NT)
	CHK_DIR_EXISTS = if not exist "$(strip $1)" mkdir "$(strip $1)"
	NUKE = rmdir /s /q
	COPY_DIR = xcopy $1 $2 /E /H /Y
	COPY = xcopy $1 $2
	FIX_PATH = $(subst /,\,$1)
	LIB_SUFFIX = .dll
else
	CHK_DIR_EXISTS = @test -d $1 || mkdir -p $1
	NUKE = rm -r $1
	COPY_DIR = cp -rv $1 $2
	COPY = cp $1 $2
	FIX_PATH = $1
	LIB_SUFFIX = .so
	REPORT = @echo -e "\e[4;1;37m$1\033[0m"
endif

CXX_FLAGS += -I/usr/include/libxml2 -g
CXX_FLAGS_RELEASE += $(CXX_FLAGS) -march=native -O2
LD_FLAGS += -lxml2 -ltiff -flto

PROJECT_DIR :=$(dir $(realpath $(lastword $(MAKEFILE_LIST))))
DEBUG_DIR := $(PROJECT_DIR)Debug/
RELEASE_DIR := $(PROJECT_DIR)Release/
SRC_DIR := $(PROJECT_DIR)
SRC_FILES := $(call rwildcard, $(SRC_DIR), *.cpp)
DEBUG_OBJ := $(patsubst $(SRC_DIR)%.cpp,$(DEBUG_DIR)%.o,$(SRC_FILES))
RELEASE_OBJ := $(patsubst $(SRC_DIR)%.cpp,$(RELEASE_DIR)%.o,$(SRC_FILES))

$(DEBUG_DIR)%.o: $(PROJECT_DIR)%.cpp
	$(call REPORT,Compiling $@ ...)
	$(call CHK_DIR_EXISTS, $(DEBUG_DIR))
	g++ -c -o $@ $(CXX_FLAGS) $<

$(RELEASE_DIR)%.o: $(PROJECT_DIR)%.cpp
	$(call REPORT,Compiling $@ ...)
	$(call CHK_DIR_EXISTS, $(RELEASE_DIR))
	g++ -c -o $@ $(CXX_FLAGS_RELEASE) $<

release: $(RELEASE_OBJ)
	$(call REPORT,Linking release build... )
	g++ $(LD_FLAGS) -o $(RELEASE_DIR)/main $(RELEASE_OBJ)

debug: $(DEBUG_OBJ)
	$(call REPORT,Linking debug build...)
	g++ $(LD_FLAGS) -o $(DEBUG_DIR)/main $(DEBUG_OBJ)

clean:
	$(call REPORT,Cleaning...)
	-$(NUKE) "$(DEBUG_DIR)"
	-$(NUKE) "$(RELEASE_DIR)"
