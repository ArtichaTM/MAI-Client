#include <sstream>
#include <iostream>

#include "tab.hpp"

#include "config.hpp"

Tab::Tab(
    const std::string& title,
    float offset,
    float height,
    sf::Color color
)
: tabText(offset, 0, title, PATH_FONT_DEFAULT, color)
{}

Tab::~Tab() {
    for (SFBase*& element : elements) {
        delete element;
    }
}

Tab::operator std::string() const {
    std::stringstream ss;
    ss << "<Tab with switcher " << (std::string) tabText << " and " << elements.size() << " elements>";
    return ss.str();
}

void Tab::draw(sf::RenderTarget& target, sf::RenderStates states) const {
    target.draw(tabText, states);
    if (!active) return;

    for (SFBase* element : elements) {
        target.draw(*element, states);
    }
}

void Tab::handleEvent(const sf::Event& event) {
    tabText.handleEvent(event);
    for (SFBase* element : elements) {
        element->handleEvent(event);
    }
}

void Tab::AddElement(SFBase* element) { elements.emplace_back(element); }

const std::string Tab::getName() { return tabText.text.getString(); }

void Tab::setActive(bool _active) {
    active = _active;
    tabText.setActive(active);
}

bool Tab::isVectorInBounds(const sf::Vector2f& vec) { return tabText.isVectorInBounds(vec); }
