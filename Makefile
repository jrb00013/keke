# Compiler and Assembler
CC = gcc
AS = nasm
LD = ld

# Compilation Flags
CFLAGS = -m32 -Wall -O2 -g -ffreestanding -nostdlib -fno-builtin
ASFLAGS = -f elf32
LDFLAGS = -T linker/linker.ld -m elf_i386

# Directories
SRC_DIR = src
BOOT_DIR = boot
INCLUDE_DIR = include
BUILD_DIR = build

# Source Files
SRC_C = $(wildcard $(SRC_DIR)/*.c)
SRC_ASM = $(wildcard $(BOOT_DIR)/*.asm)

# Object Files
OBJ_C = $(patsubst $(SRC_DIR)/%.c, $(BUILD_DIR)/%.o, $(SRC_C))
OBJ_ASM = $(patsubst $(BOOT_DIR)/%.asm, $(BUILD_DIR)/%.o, $(SRC_ASM))
OBJS = $(OBJ_C) $(OBJ_ASM)

# Output Files
KERNEL_BIN = $(BUILD_DIR)/kernel.bin
OS_BIN = $(BUILD_DIR)/os.bin

# Default Target
all: $(OS_BIN)

# Compile C source files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -I$(INCLUDE_DIR) -c $< -o $@

# Assemble assembly source files
$(BUILD_DIR)/%.o: $(BOOT_DIR)/%.asm
	@mkdir -p $(BUILD_DIR)
	$(AS) $(ASFLAGS) $< -o $@

# Link kernel binary
$(KERNEL_BIN): $(OBJS)
	$(LD) $(LDFLAGS) -o $(KERNEL_BIN) $(OBJS)

# Merge bootloader and kernel into final OS binary
$(OS_BIN): $(KERNEL_BIN)
	cat $(BOOT_DIR)/bootloader.bin $(KERNEL_BIN) > $(OS_BIN)

# Run OS in QEMU
run: $(OS_BIN)
	qemu-system-i386 -kernel $(OS_BIN)

# Clean build files
clean:
	rm -rf $(BUILD_DIR)
