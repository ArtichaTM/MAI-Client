#pragma once
#include <vector>
#include <string>

#include <SFML/Graphics.hpp>

#include "config.hpp"
#include "tab.hpp"
#include "../base.hpp"


class TabSystem : public SFBase {
    std::vector<Tab> tabs;
    size_t active_tab_index;
    sf::Font font;
    const float height;
public:
    TabSystem(float height);
    void handleEvent(const sf::Event&) override;
    void draw(sf::RenderWindow& window) const override;
    virtual void draw(sf::RenderTarget& target, sf::RenderStates states) const override;
    Tab addTab(const std::string& title);
};
