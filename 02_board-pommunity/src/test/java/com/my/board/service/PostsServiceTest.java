package com.my.board.service;

import com.my.board.application.PostsService;
import com.my.board.infrastructure.persistence.PostsRepository;
import com.my.board.domain.Role;
import com.my.board.domain.User;
import com.my.board.infrastructure.persistence.UserRepository;
import com.my.board.application.dto.PostsDto;
import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest
@Slf4j
public class PostsServiceTest {

    @Autowired
    PostsService postsService;

    @Autowired
    PostsRepository postsRepository;

    @Autowired
    UserRepository userRepository;

    @AfterEach
    public void clear() {
        postsRepository.deleteAll();
    }

    @Test
    public void 게시글_생성() {
        User user = User.builder().username("coco").nickname("coco").email("coco@coco.co").role(Role.USER).build();

        PostsDto.Request posts = PostsDto.Request.builder()
                .title("Test Title")
                .writer(user.getNickname())
                .content("Test Content")
                .view(0)
                .user(user)
                .build();

        postsService.save(posts, user.getNickname());

        log.info(String.valueOf(posts));
    }
}