#include <vector>
#include <string>
#include <sstream>
#include <cassert>
#include <iostream>

#include <SFML/Graphics.hpp>

#include "config.hpp"
#include "./system.hpp"
#include "system.hpp"


TabSystem::TabSystem(float height, sf::Color color) : height(height), color(color) {
    font.loadFromFile(PATH_FONT_DEFAULT); // Ensure you have this font file
}

TabSystem::~TabSystem() {
    for (Tab*& tab : tabs) {
        delete tab;
    }
}

TabSystem::operator std::string() const {
    std::stringstream ss;
    ss << "<TabSystem with " << tabs.size() << " tabs>";
    return ss.str();
}

void TabSystem::handleEvent(const sf::Event& event) {
    for (size_t i = 0; i < tabs.size(); ++i) {
        tabs[i]->handleEvent(event);
    }
}

void TabSystem::draw(sf::RenderTarget& target, sf::RenderStates states) const {
    assert(!tabs.empty());
    for (size_t i = 0; i < tabs.size(); ++i) {
        Tab* tab = tabs[i];
        target.draw(*tab);
    }
}

sf::FloatRect TabSystem::getGlobalBounds() const
{
    if (tabs.empty()) return sf::FloatRect();
    sf::FloatRect left_bounds = tabs[0]->getGlobalBounds();
    sf::FloatRect right_bounds = tabs[tabs.size()-1]->getGlobalBounds();
    left_bounds.width = right_bounds.left + right_bounds.width;
    return left_bounds;
}

Tab* TabSystem::addTab(const std::string& title) {
    float offset = 0;
    for (Tab*& tab : tabs) {
        offset += tab->tabText.shape.getGlobalBounds().width + offset;
    }
    Tab* tab = tabs.emplace_back(new Tab(title, offset, height, color));
    tab->tabText.setOnClick([this, tab]() {
        setActiveTab(tab);
    });
    if (tabs.size() == 1) {
        firstTabInit(tab);
    }
    return tab;
}

void TabSystem::firstTabInit(Tab* tab) {
    tab->setActive(true);
    active_tab = tab;
}

Tab* TabSystem::operator[](const std::string &name)
{
    for (Tab*& tab : tabs) {
        if (tab->getName() == name)
            return tab;
    }
    throw std::range_error("Can't find this tab");
}

Tab* TabSystem::operator[](ushort index) { return tabs[index]; }

float TabSystem::getHeight() {
    sf::FloatRect rect = tabs[tabs.size()-1]->tabText.shape.getGlobalBounds();
    return rect.top + rect.height;
}

float TabSystem::getWidth() {
    sf::FloatRect rect = tabs[tabs.size()-1]->tabText.shape.getGlobalBounds();
    return rect.left + rect.width;
}

void TabSystem::setActiveTab(Tab *tab) {
    assert(active_tab);
    active_tab->setActive(false);
    active_tab = tab;
    tab->setActive(true);
}
