#pragma once

#include <QDesktopWidget>
#include <QApplication>
#include <QRect>
#include <map>
#include <string>
#include <iostream>

class GlobalWindowLayout {
public:
	static const GlobalWindowLayout& getInstance() {

		static const GlobalWindowLayout instance;
		return instance;
	}

	const QRect getGeometry(std::string window_name) const {

		if (geometries.find(window_name) == geometries.end()) {
			std::cout << "ERROR: no global layout for this window" << std::endl;
			std::exit(0);
		}

		return geometries.at(window_name);
	}

private:
	GlobalWindowLayout() {
		QDesktopWidget* desktopWidget = QApplication::desktop();
		QRect deskRect = desktopWidget->availableGeometry();
		QRect screenRect = desktopWidget->screenGeometry();

		int screen_width = screenRect.width();
		int screen_height = screenRect.height();
		int margin_x = screen_width / 10;
		int margin_y = screen_height / 10;
		//std::cout << "screen_width: " << screen_width << std::endl;
		//std::cout << "screen_height: " << screen_height << std::endl;

		geometries[std::string("MainWindow")] = QRect(margin_x, margin_y,
																									screen_width / 3 * 2, screen_height / 3 * 2);

		geometries[std::string("ViewWindow")] = QRect(geometries[std::string("MainWindow")].x() + geometries[std::string("MainWindow")].width() + 10, geometries[std::string("MainWindow")].y(),
																									screen_width / 8, geometries[std::string("MainWindow")].height());
	}

	std::map<std::string, QRect> geometries;
};


