SHELL		:= /bin/sh
TEX			:= latexmk
TEXFLAGS	:= -bibtex -pdf -xelatex

SOURCE		:= src
SOURCES		:= $(shell find $(SOURCE) -name "*" -type f)
BUILD		:= build
FILENAME	:= szakdolgozat
TARGET		:= $(BUILD)/$(FILENAME).pdf

.PHONY: clean run

all: $(TARGET)

$(TARGET): $(SOURCES)
	@mkdir -p $(BUILD)
	@cp -r $(SOURCE)/* $(BUILD)
	@$(TEX) $(TEXFLAGS) -cd $(BUILD)/$(FILENAME).tex
	@cp $(TARGET) .

run: $(TARGET)
	xdg-open $(TARGET)

clean:
	@rm -rf $(BUILD)
