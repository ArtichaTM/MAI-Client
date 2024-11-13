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
    tabs = new HorizontalList(2., 2.);
}

TabSystem::~TabSystem() {
    delete tabs;
}

TabSystem::operator std::string() const {
    std::stringstream ss;
    ss << "<TabSystem with " << tabs->getElements().size() << " tabs>";
    return ss.str();
}

void TabSystem::handleEvent(const sf::Event& event) {
    tabs->handleEvent(event);
}

void TabSystem::draw(sf::RenderTarget& target, sf::RenderStates states) const {
    tabs->draw(target, states);
}

sf::FloatRect TabSystem::getGlobalBounds() const { return tabs->getGlobalBounds(); }

void TabSystem::move(const float left, const float top) { tabs->move(left, top); }

Tab* TabSystem::addTab(const std::string& title) {
    Tab* tab(new Tab(title, 0, 0, color));
    tabs->addElement(tab);
    tab->tabText.setOnClick([this, tab](Button* button) {
        setActiveTab(tab);
    });
    if (tabs->getElements().size() == 1) {
        tab->setActive(true);
        active_tab = tab;
    }
    return tab;
}

Tab* TabSystem::operator[](const std::string& name)
{
    for (const SFBase* _tab : tabs->getElements()) {
        Tab* tab((Tab*) _tab);
        if (tab->getName() == name)
            return tab;
    }
    throw std::range_error("Can't find this tab");
}

Tab* TabSystem::operator[](ushort index) { return (Tab*) tabs->getElements()[index]; }

void TabSystem::setActiveTab(Tab *tab) {
    assert(active_tab);
    active_tab->setActive(false);
    active_tab = tab;
    tab->setActive(true);
}

void TabSystem::fit() {
    tabs->fit();
}
