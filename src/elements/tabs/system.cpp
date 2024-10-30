#include <vector>
#include <string>
#include <iostream>

#include <SFML/Graphics.hpp>

#include "config.hpp"
#include "system.hpp"

TabSystem::TabSystem(float height) : height(height) {
    font.loadFromFile(PATH_FONT_DEFAULT); // Ensure you have this font file

    // Create tabs
    // tab.draw(*ROOT_WINDOW);
    // std::cout << "\nTab info:" << tab.tabText.getLocalBounds().width << "x" << tab.tabText.getGlobalBounds().height << '\n';
    // tabs.emplace_back("Tab 2", tab.tabText.getGlobalBounds().width, 10);
}

void TabSystem::handleEvent(const sf::Event& event) {
    sf::Vector2f localMousePos;
    if (event.type == sf::Event::MouseButtonPressed) {

        // Get the global mouse position
        sf::Vector2i globalMousePos = sf::Mouse::getPosition();

        // Convert global mouse position to window local position
        sf::Vector2f localMousePos = ROOT_WINDOW->mapPixelToCoords(globalMousePos);
    }
    for (size_t i = 0; i < tabs.size(); ++i) {
        Tab tab = tabs[i];
        if (event.type == sf::Event::MouseButtonPressed) {
            if (tab.IsInBounds(localMousePos)) {
                active_tab_index = i;
                break;
            }
        }
        tab.handleEvent(event);
    }
}

void TabSystem::draw(sf::RenderWindow& window) const {
    for (size_t i = 0; i < tabs.size(); ++i) {
        Tab tab = tabs[i];
        tab.draw(window, i == active_tab_index);
    }
    // for (size_t i = 0; i < tabs.size(); ++i) {
    //     if (i == activeTabIndex) {
    //         tabs[i].tabText.setFillColor(sf::Color::Yellow);
    //     } else {
    //         tabs[i].tabText.setFillColor(sf::Color::White);
    //     }
    //     tabs[i].draw(window);
    // }
}

void TabSystem::draw(sf::RenderTarget &target, sf::RenderStates states) const
{
}

Tab TabSystem::addTab(const std::string& title) {
    float offset = 0;
    Tab tab = tabs.emplace_back(title, offset, height);
    tab.setTabTextFont(font);
    return tab;
}
